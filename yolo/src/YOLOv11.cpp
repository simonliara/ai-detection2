#include "YOLOv11.h"

#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <map>

#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "track.h" 

extern void cuda_preprocess(const uint8_t* src, int src_w, int src_h, int src_stride,
                     float* dst_chw, int dst_w, int dst_h,
                     cudaStream_t stream);

extern void cuda_preprocess_init(int max_w, int max_h);
extern void cuda_preprocess_destroy();

extern void cuda_decode(float* predict, int num_bboxes, int num_classes, float conf_thresh, 
                        float* output, int max_objects, int* num_valid_objects, cudaStream_t stream);

extern void cuda_nms(float* output, int* num_valid_objects, float nms_thresh, int max_objects, cudaStream_t stream);

extern void cuda_decode(const float* predict, int num_bboxes, int num_classes, 
                           float conf_thresh, float* output, int max_objects, 
                           int* num_valid_objects, float scale, float ox, float oy,
                           cudaStream_t stream);
extern void cuda_nms(float* bboxes, int* num_valid_objects, float nms_thresh, 
                        int max_objects, cudaStream_t stream);

inline void checkCuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << " at " 
                  << file << ":" << line << " (" << func << ")" << std::endl;
    }
}
#define CHECK_CUDA(val) checkCuda((val), #val, __FILE__, __LINE__)

class YOLOv11::Impl {
public:
    Impl(std::string model_path, nvinfer1::ILogger& logger, float conf_thresh, float nms_thresh);
    ~Impl();
    
    std::vector<std::string> class_names;
    void loadClasses(const std::string& engine_path);

    cudaEvent_t start_event, pre_event, infer_event, post_event;
    float conf_threshold;
    float nms_threshold;
    int input_w = 0;
    int input_h = 0;
    int num_detections = 0;
    int detection_attribute_size = 0;
    int num_classes = 80;
    int last_img_w = 0;
    int last_img_h = 0;

    const int MAX_IMAGE_SIZE = 1920 * 1080;
    const int MAX_OUTPUT_BBOX_COUNT = 1000;

    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;

    float* gpu_buffers[2] = { nullptr, nullptr };
    float* gpu_output_buffer = nullptr;
    int* gpu_valid_count = nullptr;
    int* cpu_valid_count = nullptr;
    std::vector<float> cpu_buffer;
    
    std::vector<cv::Scalar> class_colors;

    void init(const std::string& engine_path, nvinfer1::ILogger& logger);
    void generateColors();
};

YOLOv11::Impl::Impl(std::string model_path, nvinfer1::ILogger& logger, float conf_thresh, float nms_thresh)
    : conf_threshold(conf_thresh), nms_threshold(nms_thresh)
{
    if (model_path.find(".engine") == std::string::npos) {
        std::cerr << "[Error] Model must be an .engine file." << std::endl;
        abort();
    }
    cpu_valid_count = new int[1];
    init(model_path, logger);
    generateColors();
}

YOLOv11::Impl::~Impl()
{
    if (stream) cudaStreamSynchronize(stream);
    
    CHECK_CUDA(cudaFree(gpu_buffers[0]));
    CHECK_CUDA(cudaFree(gpu_buffers[1]));
    CHECK_CUDA(cudaFree(gpu_output_buffer));
    CHECK_CUDA(cudaFree(gpu_valid_count));
    
    if (cpu_valid_count) delete[] cpu_valid_count;
    if (stream) cudaStreamDestroy(stream);

    cuda_preprocess_destroy();

    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;

    cudaEventDestroy(start_event);
    cudaEventDestroy(pre_event);
    cudaEventDestroy(infer_event);
    cudaEventDestroy(post_event);
}

void YOLOv11::Impl::init(const std::string& engine_path, nvinfer1::ILogger& logger)
{
    loadClasses(engine_path);

    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "[Error] Could not read engine file: " << engine_path << std::endl;
        abort();
    }

    std::streamsize size = file.tellg(); 
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "[Error] Failed to read engine file content." << std::endl;
        abort();
    }

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(buffer.data(), size);
    if (!engine) {
        std::cerr << "[Error] Failed to deserialize CUDA engine." << std::endl;
        abort();
    }
    context = engine->createExecutionContext();

#if NV_TENSORRT_MAJOR < 10
    auto in_dims = engine->getBindingDimensions(0);
    auto out_dims = engine->getBindingDimensions(1);
#else
    const char* in_name = engine->getIOTensorName(0);
    const char* out_name = engine->getIOTensorName(1);
    auto in_dims = engine->getTensorShape(in_name);
    auto out_dims = engine->getTensorShape(out_name);
#endif

    std::cout << "Output Dims: " << out_dims.d[0] << "x" << out_dims.d[1] << "x" << out_dims.d[2] << std::endl;
    input_h = in_dims.d[2];
    input_w = in_dims.d[3];

    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
        std::cout << "Tensor " << i << ": " << name << " (" << (mode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output") << ")" << std::endl;
    }
    
    detection_attribute_size = out_dims.d[1];
    num_detections = out_dims.d[2];
    num_classes = detection_attribute_size - 4;
    cpu_buffer.resize(MAX_OUTPUT_BBOX_COUNT * 7);

    CHECK_CUDA(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpu_output_buffer, MAX_OUTPUT_BBOX_COUNT * 7 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpu_valid_count, sizeof(int)));

    CHECK_CUDA(cudaStreamCreate(&stream));
    cuda_preprocess_init(1920, 1080);

    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&pre_event));
    CHECK_CUDA(cudaEventCreate(&infer_event));
    CHECK_CUDA(cudaEventCreate(&post_event));
}

void YOLOv11::Impl::loadClasses(const std::string& engine_path) {
    std::string txt_path = engine_path;
    size_t last_dot = txt_path.find_last_of(".");
    if (last_dot != std::string::npos) {
        txt_path = txt_path.substr(0, last_dot) + ".txt";
    }
    std::ifstream file(txt_path);
    if (!file.is_open() && last_dot != std::string::npos) {
        std::string fallback_path = txt_path;
        size_t prev_dot = fallback_path.find_last_of(".", last_dot - 1);
        if (prev_dot != std::string::npos) {
            fallback_path = fallback_path.substr(0, prev_dot) + ".txt";
            file.open(fallback_path); 
            if (file.is_open()) {
                txt_path = fallback_path;
            }
        }
    }

    if (!file.is_open()) {
        std::cerr << "[WARN] Class file not found (tried: " << txt_path << ")!\n";
        exit(1);
    }

    std::cout << "[INFO] Loading classes from: " << txt_path << std::endl;
    std::string line;
    class_names.clear();
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        class_names.push_back(line);
    }
    std::cout << "[INFO] Loaded " << class_names.size() << " classes.\n";
}

void YOLOv11::Impl::generateColors() {
    cv::RNG rng(12345);
    for(int i=0; i<num_classes; ++i) {
        class_colors.emplace_back(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }
}

YOLOv11::YOLOv11(std::string model_path, nvinfer1::ILogger& logger, float conf_thresh, float nms_thresh)
    : pImpl(std::make_unique<Impl>(model_path, logger, conf_thresh, nms_thresh)) 
{
    // for(int i=0; i<5; ++i) this->infer();
}

YOLOv11::~YOLOv11() = default;

float YOLOv11::getConfThreshold() const { return pImpl->conf_threshold; }
float YOLOv11::getNMSThreshold() const { return pImpl->nms_threshold; }

void YOLOv11::preprocess(cv::Mat& image) {
    cudaEventRecord(pImpl->start_event, pImpl->stream);
    if (image.empty()) return;

    cv::Mat bgr;
    if (image.channels() == 3 && image.type() == CV_8UC3) {
        bgr = image;
    } else if (image.channels() == 4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        std::cerr << "Unsupported input type\n";
        return;
    }

    // If step != cols*3, memcpy2DAsync handles it, so clone is optional now.
    // But it doesn’t hurt to keep:
    // if (!bgr.isContinuous()) bgr = bgr.clone();

    pImpl->last_img_w = bgr.cols;
    pImpl->last_img_h = bgr.rows;

    cuda_preprocess(bgr.data, bgr.cols, bgr.rows, (int)bgr.step,
                    pImpl->gpu_buffers[0], pImpl->input_w, pImpl->input_h,
                    pImpl->stream);

    // CHECK_CUDA(cudaStreamSynchronize(pImpl->stream));
    cudaEventRecord(pImpl->pre_event, pImpl->stream);
}

void YOLOv11::infer()
{
    const char* input_name = pImpl->engine->getIOTensorName(0);
    const char* output_name = pImpl->engine->getIOTensorName(1);

    pImpl->context->setTensorAddress(input_name, pImpl->gpu_buffers[0]);
    pImpl->context->setTensorAddress(output_name, pImpl->gpu_buffers[1]);
    pImpl->context->enqueueV3(pImpl->stream);
    cudaEventRecord(pImpl->infer_event, pImpl->stream);
}

void YOLOv11::postprocess(std::vector<YoloDetection>& output)
{
    output.clear();
    
    float r_w = pImpl->input_w / (float)pImpl->last_img_w;
    float r_h = pImpl->input_h / (float)pImpl->last_img_h;
    float scale = (r_w < r_h) ? r_w : r_h;
    float offset_x = (pImpl->input_w - pImpl->last_img_w * scale) / 2.0f;
    float offset_y = (pImpl->input_h - pImpl->last_img_h * scale) / 2.0f;

    CHECK_CUDA(cudaMemsetAsync(pImpl->gpu_valid_count, 0, sizeof(int), pImpl->stream));
    cuda_decode(
        pImpl->gpu_buffers[1], pImpl->num_detections, pImpl->num_classes, pImpl->conf_threshold,
        pImpl->gpu_output_buffer, pImpl->MAX_OUTPUT_BBOX_COUNT, pImpl->gpu_valid_count,
        scale, offset_x, offset_y, pImpl->stream);
    cuda_nms(
        pImpl->gpu_output_buffer, pImpl->gpu_valid_count, pImpl->nms_threshold, 
        pImpl->MAX_OUTPUT_BBOX_COUNT, pImpl->stream);
    CHECK_CUDA(cudaMemcpyAsync(pImpl->cpu_valid_count, pImpl->gpu_valid_count, sizeof(int), 
                               cudaMemcpyDeviceToHost, pImpl->stream));
    
    CHECK_CUDA(cudaMemcpyAsync(pImpl->cpu_buffer.data(), pImpl->gpu_output_buffer, 
                                pImpl->MAX_OUTPUT_BBOX_COUNT * 7 * sizeof(float), 
                                cudaMemcpyDeviceToHost, pImpl->stream));
    cudaEventRecord(pImpl->post_event, pImpl->stream);
    CHECK_CUDA(cudaStreamSynchronize(pImpl->stream));

    int num_candidates = std::min(pImpl->cpu_valid_count[0], pImpl->MAX_OUTPUT_BBOX_COUNT);

    for (int i = 0; i < num_candidates; ++i) {
        float* item = &pImpl->cpu_buffer[i * 7];
        if (item[6] < 0.5f) continue; 

        YoloDetection res;
        res.class_id = (int)item[5];
        res.conf = item[4];
        
        int x = (int)item[0];
        int y = (int)item[1];
        int w = (int)item[2] - x;
        int h = (int)item[3] - y;

        x = std::max(0, std::min(x, pImpl->last_img_w - 1));
        y = std::max(0, std::min(y, pImpl->last_img_h - 1));
        w = std::max(0, std::min(w, pImpl->last_img_w - x));
        h = std::max(0, std::min(h, pImpl->last_img_h - y));

        if (w > 0 && h > 0) {
            res.bbox = cv::Rect(x, y, w, h);
            output.push_back(res);
        }
    }
}

YoloStats YOLOv11::getStats() const {
    YoloStats stats;
    cudaEventElapsedTime(&stats.pre,   pImpl->start_event, pImpl->pre_event);
    cudaEventElapsedTime(&stats.infer, pImpl->pre_event,   pImpl->infer_event);
    cudaEventElapsedTime(&stats.post,  pImpl->infer_event, pImpl->post_event);
    return stats;
}

std::vector<byte_track::Object> YOLOv11::toByteTrackObjects(
    const std::vector<YoloDetection>& yolo,
    float conf_thresh
) {
    std::vector<byte_track::Object> out;
    out.reserve(yolo.size());

    for (const auto& d : yolo) {
        if (d.conf < conf_thresh) continue;

        out.emplace_back(
            byte_track::Rect<float>(
                static_cast<float>(d.bbox.x),
                static_cast<float>(d.bbox.y),
                static_cast<float>(d.bbox.width),
                static_cast<float>(d.bbox.height)
            ),
            d.class_id,
            d.conf
        );
    }
    return out;
}

void YOLOv11::draw(cv::Mat& image, const std::vector<YoloDetection>& output)
{
    for (const auto& det : output) {
        cv::Scalar color = pImpl->class_colors[det.class_id % pImpl->class_colors.size()];
        cv::rectangle(image, det.bbox, color, 2);

        std::string name = (det.class_id >= 0 && det.class_id < pImpl->class_names.size()) 
                           ? pImpl->class_names[det.class_id] 
                           : std::to_string(det.class_id);
                           
        std::string label = name + " " + cv::format("%.2f", det.conf);
        
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        int top = std::max(det.bbox.y, labelSize.height);
        cv::rectangle(image, cv::Point(det.bbox.x, top - labelSize.height),
                      cv::Point(det.bbox.x + labelSize.width, top + baseLine), color, cv::FILLED);
        cv::putText(image, label, cv::Point(det.bbox.x, top), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    }
}

void YOLOv11::draw(cv::Mat& image, const std::vector<std::shared_ptr<Track>>& tracks)
{
    static std::map<int, cv::Scalar> track_colors;
    
    for (const auto& t : tracks) {
        if (!t) continue;
        
        int id = t->track_id; 
        std::vector<float> tlwh = t->get_tlwh();

        if (track_colors.find(id) == track_colors.end()) {
            cv::RNG rng(id * 99); 
            track_colors[id] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        }

        cv::Rect box((int)tlwh[0], (int)tlwh[1], (int)tlwh[2], (int)tlwh[3]);
        
        box.x = std::max(0, std::min(box.x, image.cols - 1));
        box.y = std::max(0, std::min(box.y, image.rows - 1));
        box.width = std::min(box.width, image.cols - box.x);
        box.height = std::min(box.height, image.rows - box.y);

        if(box.width <= 0 || box.height <= 0) continue;

        cv::rectangle(image, box, track_colors[id], 2);

        std::string label = "ID: " + std::to_string(id);
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        
        cv::Rect label_bg(box.x, std::max(0, box.y - labelSize.height - 5), 
                          labelSize.width + 4, labelSize.height + 4);
                          
        cv::rectangle(image, label_bg, track_colors[id], cv::FILLED);
        cv::putText(image, label, cv::Point(box.x + 2, label_bg.y + labelSize.height), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);
    }
}

void YOLOv11::draw(cv::Mat& image, const std::vector<std::shared_ptr<byte_track::STrack>>& tracks)
{
    static std::map<int, cv::Scalar> track_colors;
    
    for (const auto& t : tracks) {
        if (!t) continue;
        
        int id = t->getTrackId(); 
        int classId = t->getClassId();
        const auto& rect = t->getRect();

        if (track_colors.find(id) == track_colors.end()) {
            cv::RNG rng(id * 99); 
            track_colors[id] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        }

        cv::Rect box((int)rect.x(), (int)rect.y(), (int)rect.width(), (int)rect.height());
        box &= cv::Rect(0, 0, image.cols, image.rows);

        if(box.width <= 0 || box.height <= 0) continue;

        cv::rectangle(image, box, track_colors[id], 2);

        std::string className = (classId >= 0 && classId < pImpl->class_names.size()) 
                                ? pImpl->class_names[classId] 
                                : "ID: " + std::to_string(id);
        
        std::string label = className + " #" + std::to_string(id);
        
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        int label_y = std::max(box.y, labelSize.height + 5);
        cv::Rect label_bg(box.x, label_y - labelSize.height - 5, 
                          labelSize.width + 4, labelSize.height + 4);
                          
        cv::rectangle(image, label_bg, track_colors[id], cv::FILLED);
        cv::putText(image, label, cv::Point(box.x + 2, label_y - 2), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}