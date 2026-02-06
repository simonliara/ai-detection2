#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <thread>

#include <cxxopts.hpp>

#include <dds/dds.hpp>
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

#include <data_bus/topics/sensor_data.h>
#include <data_bus/topics/perception.h>

#include <DataBus/SensorData/Image.hpp>
#include <DataBus/Perception/ObjectDetection.hpp>

#include "YOLOv11.h"
#include "BoTSORT.h"
#include "DataType.h"
#include "track.h"

#include "ByteTrack/BYTETracker.h"
#include "ByteTrack/Object.h"

static std::atomic<bool> g_running{true};
static void onSignal(int) { g_running = false; }

static inline cv::Rect clampRect(const cv::Rect& r, int W, int H) {
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int w = std::min(r.width,  W - x);
    int h = std::min(r.height, H - y);
    if (w <= 0 || h <= 0) return cv::Rect();
    return cv::Rect(x, y, w, h);
}

static inline double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

static inline void rectToXywhn(const cv::Rect& boxPx, int imgW, int imgH,
                              double& cx, double& cy, double& sx, double& sy) {
    const double x = (double)boxPx.x;
    const double y = (double)boxPx.y;
    const double w = (double)boxPx.width;
    const double h = (double)boxPx.height;

    cx = clamp01((x + 0.5 * w) / (double)imgW);
    cy = clamp01((y + 0.5 * h) / (double)imgH);
    sx = clamp01(w / (double)imgW);
    sy = clamp01(h / (double)imgH);
}

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
} g_trtLogger;

static void setCycloneEnvOnce(const std::string& iface,
                              bool allowMulticast,
                              bool enableMulticastLoopback) {
    std::string xml;
    xml += "<CycloneDDS>";
    xml +=   "<Domain>";
    xml +=     "<General>";
    xml +=       "<Interfaces>";
    xml +=         "<NetworkInterface address=\"" + iface + "\"/>";
    xml +=       "</Interfaces>";
    xml +=       "<AllowMulticast>" + std::string(allowMulticast ? "true" : "false") + "</AllowMulticast>";
    xml +=       "<EnableMulticastLoopback>" + std::string(enableMulticastLoopback ? "true" : "false") + "</EnableMulticastLoopback>";
    xml +=     "</General>";
    xml +=   "</Domain>";
    xml += "</CycloneDDS>";

#if defined(_WIN32)
    _putenv_s("CYCLONEDDS_URI", xml.c_str());
#else
    ::setenv("CYCLONEDDS_URI", xml.c_str(), 1);
#endif
}

struct AppConfig {
    int domainId = 0;
    std::string iface = "127.0.0.1";
    bool allowMulticast = true;
    bool enableMulticastLoopback = true;

    std::string yoloEngine = "ai_detection_cxx/yolo/weights/yolo11s_static.fp16.engine";
    float confThresh = 0.1f;
    float nmsThresh  = 0.7f;

    std::string trackerIni = "ai_detection_cxx/botsort/config/tracker.ini";
    std::string gmcIni     = "ai_detection_cxx/botsort/config/gmc.ini";
    std::string reidIni    = "ai_detection_cxx/botsort/config/reid.ini";
    std::string reidOnnx   = "ai_detection_cxx/botsort/weights/osnet_x0_25_market1501.onnx";
    std::string trackerType = "bytetrack"; // "bytetrack" or "botsort"
    bool verbose = false;
};

class DataBusDetectorNode {
public:
    explicit DataBusDetectorNode(const AppConfig& cfg)
    : cfg_(cfg),
      participant_(cfg_.domainId),
      imageReader_(DataBus::Topics::SensorData::getImagesReader(participant_)),
      detWriter_(DataBus::Topics::Perception::getObjectDetectionWriter(participant_)),
      model_(cfg_.yoloEngine, g_trtLogger, cfg_.confThresh, cfg_.nmsThresh)
    {
        use_botsort_ = (cfg_.trackerType == "botsort");

        if (use_botsort_) {
            botsort_ = std::make_unique<BoTSORT>(cfg_.trackerIni, cfg_.gmcIni, cfg_.reidIni, cfg_.reidOnnx);
        } else {
            tracker_ = std::make_unique<byte_track::BYTETracker>();
        }
    }

    void loop() {
        while (g_running) {
            bool gotAny = false;

            auto t1 = std::chrono::high_resolution_clock::now();
            auto samples = imageReader_.take();
            for (auto& s : samples) {
                gotAny = true;
                if (!s.info().valid()) continue;

                const auto& timed = s.data();
                auto imageOpt = decodeTimedImage(timed);
                if (!imageOpt.has_value()) continue;

                processFrame(*imageOpt, timed);
            }

            if (!gotAny) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            } else {
                auto t2 = std::chrono::high_resolution_clock::now();
                double loop_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                spdlog::info("Loop time: {} ms", loop_ms);
            }
        }
    }

private:
    void processFrame(const cv::Mat& bgr, const DataBus::SensorData::TimedImage& timedImage) {
        if (bgr.empty()) return;
        auto ta = std::chrono::high_resolution_clock::now();

        std::vector<YoloDetection> yoloDetections;
        auto t0 = std::chrono::high_resolution_clock::now();
        model_.preprocess(bgr);
        model_.infer();
        model_.postprocess(yoloDetections);
        auto t1 = std::chrono::high_resolution_clock::now();

        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<std::shared_ptr<byte_track::STrack>> tracks;

        if (!use_botsort_) {
            auto inputs = YOLOv11::toByteTrackObjects(yoloDetections, cfg_.confThresh);
            auto tUpdate0 = std::chrono::high_resolution_clock::now();
            tracks = tracker_->update(inputs);
            auto tUpdate1 = std::chrono::high_resolution_clock::now();
            (void)tUpdate0;
            (void)tUpdate1;
        } else {
            
        }

        auto t3 = std::chrono::high_resolution_clock::now();

        for (const auto& tp : tracks) {
            if (!tp) continue;
            const int trackId = static_cast<int>(tp->getTrackId());
            const auto& rect  = tp->getRect();

            cv::Rect boxPx = clampRect(cv::Rect(rect.x(), rect.y(), rect.width(), rect.height()), bgr.cols, bgr.rows);
            if (boxPx.area() <= 0) continue;

            int classId = tp->getClassId();
            std::string className =
                (classId >= 0 && classId < (int)model_.getClassNames().size()) ? model_.getClassNames()[classId] : "unknown";

            auto it = trackClass_.find(trackId);
            if (it == trackClass_.end()) {
                if (className == "unknown") continue;
                trackClass_[trackId] = className;
            } else {
                className = it->second;
            }

            double cx, cy, sx, sy;
            rectToXywhn(boxPx, bgr.cols, bgr.rows, cx, cy, sx, sy);

            DataBus::Perception::ObjectDetection msg;
            msg.id((uint64_t)trackId);
            msg.sourceIdentity(timedImage.sourceIdentity());
            msg.time(timedImage.imageTime());
            msg.classification(className);
            msg.confidenceScore(tp->getScore());
            msg.centerX(cx);
            msg.centerY(cy);
            msg.sizeX(sx);
            msg.sizeY(sy);
            detWriter_.write(msg);
        }

        cv::Mat vizImage = bgr.clone();
        model_.draw(vizImage, tracks);

        auto tb = std::chrono::high_resolution_clock::now();

        double yolo_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double track_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        spdlog::info("Inference: {:.2f}ms | Track: {:.2f}ms | Total: {:.2f}ms",
                     yolo_ms, track_ms,
                     std::chrono::duration<double, std::milli>(tb - ta).count());

        // static int frame_idx = 0;
        // cv::imwrite("output/frame_" + std::to_string(++frame_idx) + ".jpg", vizImage);
    }

    std::optional<cv::Mat> decodeTimedImage(const DataBus::SensorData::TimedImage& msg) {
        const auto& im = msg.image();
        const auto& data = im.data();
        if (data.empty()) return std::nullopt;

        if (im.encoding() == DataBus::SensorData::Encoding::BGR24) {
            const int H = static_cast<int>(im.height());
            const int W = static_cast<int>(im.width());
            if (data.size() < static_cast<size_t>(H * W * 3)) return std::nullopt;
            return cv::Mat(H, W, CV_8UC3,
                           const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(data.data())));
        }

        if (im.encoding() == DataBus::SensorData::Encoding::JPEG) {
            cv::Mat rawData(1, static_cast<int>(data.size()), CV_8UC1, (void*)data.data());
            cv::Mat decoded = cv::imdecode(rawData, cv::IMREAD_COLOR);
            if (decoded.empty()) return std::nullopt;
            return decoded;
        }

        return std::nullopt;
    }

private:
    AppConfig cfg_;

    dds::domain::DomainParticipant participant_;
    DataBus::Topics::SensorData::ImagesReader imageReader_;
    DataBus::Topics::Perception::ObjectDetectionWriter detWriter_;

    YOLOv11 model_;

    std::unique_ptr<BoTSORT> botsort_;
    bool use_botsort_ = false;

    std::unique_ptr<byte_track::BYTETracker> tracker_;
    byte_track::BYTETracker fallbackByteTracker_; // used only if trackerType==botsort (placeholder behavior)

    std::unordered_map<int, std::string> trackClass_;
    cv::Mat jpegScratchBuffer_;

    static constexpr bool kShowUi = true;
    static constexpr const char* kWinName = "yolo+tracking";
};

int main(int argc, char* argv[]) {
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    cxxopts::Options options("DataBus detector");
    // clang-format off
    options.add_options()
        ("h,help", "Show help")
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))

        ("domain-id", "DataBus domain id", cxxopts::value<int>())
        ("iface", "CycloneDDS interface address", cxxopts::value<std::string>()->default_value("127.0.0.1"))
        ("allow-multicast", "CycloneDDS AllowMulticast", cxxopts::value<bool>()->default_value("true"))
        ("enable-multicast-loopback", "CycloneDDS EnableMulticastLoopback", cxxopts::value<bool>()->default_value("true"))

        ("yolo-engine", "Path to TensorRT engine", cxxopts::value<std::string>()->default_value("ai_detection_cxx/yolo/weights/yolo11s_static.fp16.engine"))
        ("conf-thresh", "YOLO confidence threshold", cxxopts::value<float>()->default_value("0.1"))
        ("nms-thresh", "YOLO NMS threshold", cxxopts::value<float>()->default_value("0.7"))

        ("tracker-type", "Tracker: bytetrack or botsort", cxxopts::value<std::string>()->default_value("bytetrack"))
        ("tracker-ini", "BoTSORT tracker.ini", cxxopts::value<std::string>()->default_value("ai_detection_cxx/botsort/config/tracker.ini"))
        ("gmc-ini", "BoTSORT gmc.ini", cxxopts::value<std::string>()->default_value("ai_detection_cxx/botsort/config/gmc.ini"))
        ("reid-ini", "BoTSORT reid.ini", cxxopts::value<std::string>()->default_value("ai_detection_cxx/botsort/config/reid.ini"))
        ("reid-onnx", "BoTSORT ReID ONNX", cxxopts::value<std::string>()->default_value("ai_detection_cxx/botsort/weights/osnet_x0_25_market1501.onnx"));
    // clang-format on

    cxxopts::ParseResult args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!args.count("domain-id")) {
        std::cout << "Missing argument --domain-id" << std::endl;
        return 1;
    }


    AppConfig cfg;
    cfg.verbose = args["verbose"].as<bool>();

    cfg.domainId = args["domain-id"].as<int>();
    cfg.iface = args["iface"].as<std::string>();
    cfg.allowMulticast = args["allow-multicast"].as<bool>();
    cfg.enableMulticastLoopback = args["enable-multicast-loopback"].as<bool>();

    cfg.yoloEngine = args["yolo-engine"].as<std::string>();
    cfg.confThresh = args["conf-thresh"].as<float>();
    cfg.nmsThresh  = args["nms-thresh"].as<float>();

    cfg.trackerType = args["tracker-type"].as<std::string>();
    cfg.trackerIni  = args["tracker-ini"].as<std::string>();
    cfg.gmcIni      = args["gmc-ini"].as<std::string>();
    cfg.reidIni     = args["reid-ini"].as<std::string>();
    cfg.reidOnnx    = args["reid-onnx"].as<std::string>();

    spdlog::set_level(spdlog::level::info);
    if (cfg.verbose) spdlog::set_level(spdlog::level::debug);

    setCycloneEnvOnce(cfg.iface, cfg.allowMulticast, cfg.enableMulticastLoopback);

    spdlog::info("Starting DataBus detector (domain={}, iface={})", cfg.domainId, cfg.iface);
    spdlog::info("YOLO engine: {}", cfg.yoloEngine);
    spdlog::info("YOLO conf_thresh: {} | nms_thresh: {}", cfg.confThresh, cfg.nmsThresh);
    spdlog::info("Tracker type: {}", cfg.trackerType);
    spdlog::info("BoTSORT cfg: {}, {}, {}", cfg.trackerIni, cfg.gmcIni, cfg.reidIni);
    spdlog::info("BoTSORT reid: {}", cfg.reidOnnx);

    DataBusDetectorNode node(cfg);
    node.loop();

    spdlog::info("Exiting");
    return 0;
}
