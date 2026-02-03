#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <spdlog/spdlog.h>

static inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
}

class TrtLogger : public nvinfer1::ILogger {
public:
    explicit TrtLogger(Severity s = Severity::kINFO) : sev_(s) {}
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= sev_) {
            switch (severity) {
                case Severity::kINTERNAL_ERROR:
                case Severity::kERROR:
                    spdlog::error("[TRT] {}", msg);
                    break;
                case Severity::kWARNING:
                    spdlog::warn("[TRT] {}", msg);
                    break;
                case Severity::kINFO:
                    spdlog::info("[TRT] {}", msg);
                    break;
                case Severity::kVERBOSE:
                    spdlog::debug("[TRT] {}", msg);
                    break;
                default:
                    spdlog::info("[TRT] {}", msg);
                    break;
            }
        }
    }
private:
    Severity sev_;
};

template <typename T>
struct TrtDestroy {
    void operator()(T* p) const noexcept {
        if (p) {
#if NV_TENSORRT_MAJOR < 10
            p->destroy();
#else
            delete p;
#endif
        }
    }
};

template <typename T>
using TrtUnique = std::unique_ptr<T, TrtDestroy<T>>;

enum class Precision { FP32, FP16, INT8 };

static Precision parsePrecision(const std::string& s) {
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return std::tolower(c); });
    if (t == "fp16") return Precision::FP16;
    if (t == "int8") return Precision::INT8;
    return Precision::FP32;
}

static const char* precisionName(Precision p) {
    switch (p) {
        case Precision::FP16: return "fp16";
        case Precision::INT8: return "int8";
        default: return "fp32";
    }
}

static bool fileExists(const std::string& p) {
    std::ifstream f(p);
    return f.good();
}

static void writeFile(const std::string& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open output file: " + path);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!f.good()) throw std::runtime_error("Failed to write engine file: " + path);
}

static bool hasDynamicDims(const nvinfer1::Dims& d) {
    for (int i = 0; i < d.nbDims; ++i) if (d.d[i] == -1) return true;
    return false;
}

static std::string stemNoExt(const std::string& path) {
    std::string base = path;
    auto slash = base.find_last_of("/\\");
    if (slash != std::string::npos) base = base.substr(slash + 1);

    auto dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);

    std::string dir;
    if (slash != std::string::npos) dir = path.substr(0, slash + 1);
    return dir + base;
}

static void parseHxW(const std::string& s, int& h, int& w) {
    auto x = s.find_first_of("xX");
    if (x == std::string::npos) throw std::runtime_error("Bad HxW: " + s);
    h = std::stoi(s.substr(0, x));
    w = std::stoi(s.substr(x + 1));
    if (h <= 0 || w <= 0) throw std::runtime_error("Bad HxW: " + s);
}

static void usage() {
    std::cerr <<
        "Usage:\n"
        "  onnx2engine <model.onnx> [fp32|fp16|int8] [gpu_id] [workspace_mb] [verbose]\n"
        "            [--batch N] [--minhw HxW] [--opthw HxW] [--maxhw HxW]\n\n"
        "Output:\n"
        "  Writes <model_stem>.<precision>.engine next to the ONNX.\n\n"
        "Notes:\n"
        "  - Fixed-shape ONNX: builds a fixed engine.\n"
        "  - Dynamic-shape ONNX: auto-creates an optimization profile.\n"
        "    Defaults: min=640x640 opt=960x960 max=1280x1280, batch=1.\n\n"
        "Examples:\n"
        "  onnx2engine yolo11s.onnx fp16 0 2048\n"
        "  onnx2engine yolo_dyn.onnx fp16 0 4096 0 --minhw 640x640 --opthw 960x960 --maxhw 1280x1280\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 2; }

        const std::string onnxPath = argv[1];
        const Precision precision  = (argc >= 3) ? parsePrecision(argv[2]) : Precision::FP16;
        const int gpuId            = (argc >= 4) ? std::stoi(argv[3]) : 0;
        const size_t workspaceMB   = (argc >= 5) ? static_cast<size_t>(std::stoul(argv[4])) : 2048;
        const bool verbose         = (argc >= 6) ? (std::stoi(argv[5]) != 0) : false;

        int batch = 1;
        int minH = 640, minW = 640;
        int optH = 960, optW = 960;
        int maxH = 1280, maxW = 1280;

        for (int i = 6; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--batch" && i + 1 < argc) batch = std::stoi(argv[++i]);
            else if (a == "--minhw" && i + 1 < argc) parseHxW(argv[++i], minH, minW);
            else if (a == "--opthw" && i + 1 < argc) parseHxW(argv[++i], optH, optW);
            else if (a == "--maxhw" && i + 1 < argc) parseHxW(argv[++i], maxH, maxW);
            else throw std::runtime_error("Unknown arg: " + a);
        }

        if (!fileExists(onnxPath)) throw std::runtime_error("ONNX file not found: " + onnxPath);
        if (batch <= 0) throw std::runtime_error("batch must be > 0");
        if (!(minH <= optH && optH <= maxH && minW <= optW && optW <= maxW))
            throw std::runtime_error("Require min<=opt<=max for both H and W");

        const std::string outPath = stemNoExt(onnxPath) + "." + precisionName(precision) + ".engine";

        checkCuda(cudaSetDevice(gpuId), "cudaSetDevice failed");

        TrtLogger logger(verbose ? nvinfer1::ILogger::Severity::kVERBOSE
                                 : nvinfer1::ILogger::Severity::kINFO);

        TrtUnique<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        if (!builder) throw std::runtime_error("createInferBuilder failed");

#if NV_TENSORRT_MAJOR < 10
        const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        TrtUnique<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
#else
        TrtUnique<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
#endif

        TrtUnique<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config) throw std::runtime_error("createBuilderConfig failed");

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspaceMB * (1ULL << 20));

        if (precision == Precision::FP16) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        } else if (precision == Precision::INT8) {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            std::cerr << "[WARN] INT8 selected. You need Q/DQ ONNX or an INT8 calibrator.\n";
        }
        
        TrtUnique<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        if (!parser) throw std::runtime_error("createParser failed");

        int parseVerbosity = verbose ? (int)nvinfer1::ILogger::Severity::kVERBOSE
                                     : (int)nvinfer1::ILogger::Severity::kINFO;

        if (!parser->parseFromFile(onnxPath.c_str(), parseVerbosity)) {
            std::cerr << "ONNX parse failed.\n";
            for (int i = 0; i < parser->getNbErrors(); ++i)
                std::cerr << "  [ONNX ERROR] " << parser->getError(i)->desc() << "\n";
            return 1;
        }

        bool needProfile = false;
        for (int i = 0; i < network->getNbInputs(); ++i) {
            auto d = network->getInput(i)->getDimensions();
            if (hasDynamicDims(d)) needProfile = true;
        }

        if (needProfile) {
            auto profile = builder->createOptimizationProfile();
            if (!profile) throw std::runtime_error("createOptimizationProfile failed");

            for (int i = 0; i < network->getNbInputs(); ++i) {
                auto* inp = network->getInput(i);
                const char* name = inp->getName();
                auto model = inp->getDimensions();

                if (model.nbDims != 4) {
                    throw std::runtime_error(std::string("Dynamic input rank != 4 not supported for auto-profile on input: ") + name);
                }

                auto make = [&](int N, int H, int W) {
                    nvinfer1::Dims d = model;
                    if (d.d[0] == -1) d.d[0] = N;
                    if (d.d[2] == -1) d.d[2] = H;
                    if (d.d[3] == -1) d.d[3] = W;
                    if (d.d[1] == -1) d.d[1] = 3;
                    return d;
                };

                nvinfer1::Dims dMin = make(batch, minH, minW);
                nMin:;

                nvinfer1::Dims dOpt = make(batch, optH, optW);
                nvinfer1::Dims dMax = make(batch, maxH, maxW);

                for (int k = 0; k < model.nbDims; ++k) {
                    if (dMin.d[k] == -1 || dOpt.d[k] == -1 || dMax.d[k] == -1)
                        throw std::runtime_error(std::string("Auto-profile could not resolve -1 for input: ") + name);
                }

                if (!profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, dMin) ||
                    !profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, dOpt) ||
                    !profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, dMax)) {
                    throw std::runtime_error(std::string("setDimensions failed for input: ") + name);
                }

                spdlog::info("Profile for {} MIN=[{},{},{},{}] OPT=[{},{},{},{}] MAX=[{},{},{},{}]",
                    name,
                    dMin.d[0], dMin.d[1], dMin.d[2], dMin.d[3],
                    dOpt.d[0], dOpt.d[1], dOpt.d[2], dOpt.d[3],
                    dMax.d[0], dMax.d[1], dMax.d[2], dMax.d[3]);
            }

            config->addOptimizationProfile(profile);
        } else {
            for (int i = 0; i < network->getNbInputs(); ++i) {
                auto d = network->getInput(i)->getDimensions();
                spdlog::debug("Static input {} dims=[{},{},{},{}]",
                    network->getInput(i)->getName(),
                    d.d[0], d.d[1], d.d[2], d.d[3]);
            }
        }

        TrtUnique<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        if (!plan) throw std::runtime_error("buildSerializedNetwork failed (plan is null)");

        writeFile(outPath, plan->data(), plan->size());

        spdlog::info("Wrote engine: {} (precision={}, gpu={}, workspaceMB={}, dynamic={})",
            outPath,
            precisionName(precision),
            gpuId,
            workspaceMB,
            needProfile ? "yes" : "no");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        usage();
        return 1;
    }
}
