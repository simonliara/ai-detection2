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

static const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

extern void cuda_preprocess(uint8_t* src, int src_width, int src_height, 
                           float* dst, int dst_width, int dst_height, 
                           cudaStream_t stream);

extern void cuda_preprocess_init(int max_image_size);
extern void cuda_preprocess_destroy();

extern void cuda_decode(float* predict, int num_bboxes, int num_classes, float conf_thresh, 
                        float* output, int max_objects, int* num_valid_objects, cudaStream_t stream);

extern void cuda_nms(float* output, int* num_valid_objects, float nms_thresh, int max_objects, cudaStream_t stream);

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

    float conf_threshold;
    float nms_threshold;
    int input_w = 0;
    int input_h = 0;
    int num_detections = 0;
    int detection_attribute_size = 0;
    int num_classes = 80;
    int last_img_w = 0;
    int last_img_h = 0;

    const int MAX_IMAGE_SIZE = 4096 * 4096;
    const int MAX_OUTPUT_BBOX_COUNT = 1000;

    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;

    float* gpu_buffers[2] = { nullptr, nullptr };
    float* gpu_output_buffer = nullptr;
    int* gpu_valid_count = nullptr;
    int* cpu_valid_count = nullptr;
    
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
}

void YOLOv11::Impl::init(const std::string& engine_path, nvinfer1::ILogger& logger)
{
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "[Error] Could not read engine file." << std::endl;
        abort();
    }

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(buffer.data(), size);
    context = engine->createExecutionContext();

    auto in_dims = engine->getBindingDimensions(0);
    input_h = in_dims.d[2];
    input_w = in_dims.d[3];
    
    auto out_dims = engine->getBindingDimensions(1);
    detection_attribute_size = out_dims.d[1];
    num_detections = out_dims.d[2];
    num_classes = detection_attribute_size - 4;

    CHECK_CUDA(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpu_output_buffer, MAX_OUTPUT_BBOX_COUNT * 6 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpu_valid_count, sizeof(int)));

    CHECK_CUDA(cudaStreamCreate(&stream));
    cuda_preprocess_init(MAX_IMAGE_SIZE);
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
    for(int i=0; i<5; ++i) this->infer();
}

YOLOv11::~YOLOv11() = default;

float YOLOv11::getConfThreshold() const { return pImpl->conf_threshold; }
float YOLOv11::getNMSThreshold() const { return pImpl->nms_threshold; }

void YOLOv11::preprocess(cv::Mat& image)
{
    if (image.empty()) return;
    pImpl->last_img_w = image.cols;
    pImpl->last_img_h = image.rows;

    cv::Mat input_mat;
    if (image.type() == CV_8UC3) input_mat = image;
    else if (image.channels() == 1) cv::cvtColor(image, input_mat, cv::COLOR_GRAY2BGR);
    else if (image.channels() == 4) cv::cvtColor(image, input_mat, cv::COLOR_BGRA2BGR);
    else image.copyTo(input_mat); 

    cuda_preprocess(input_mat.data, input_mat.cols, input_mat.rows, 
                    pImpl->gpu_buffers[0], pImpl->input_w, pImpl->input_h, pImpl->stream);
    CHECK_CUDA(cudaStreamSynchronize(pImpl->stream));
}

void YOLOv11::infer()
{
    pImpl->context->enqueueV2((void**)pImpl->gpu_buffers, pImpl->stream, nullptr);
}

void YOLOv11::postprocess(std::vector<YoloDetection>& output)
{
    CHECK_CUDA(cudaMemsetAsync(pImpl->gpu_valid_count, 0, sizeof(int), pImpl->stream));
    cuda_decode(pImpl->gpu_buffers[1], pImpl->num_detections, pImpl->num_classes, pImpl->conf_threshold,
                pImpl->gpu_output_buffer, pImpl->MAX_OUTPUT_BBOX_COUNT, pImpl->gpu_valid_count,
                pImpl->stream);

    cuda_nms(pImpl->gpu_output_buffer, pImpl->gpu_valid_count, pImpl->nms_threshold, 
             pImpl->MAX_OUTPUT_BBOX_COUNT, pImpl->stream);

    CHECK_CUDA(cudaMemcpyAsync(pImpl->cpu_valid_count, pImpl->gpu_valid_count, sizeof(int), 
                               cudaMemcpyDeviceToHost, pImpl->stream));
    CHECK_CUDA(cudaStreamSynchronize(pImpl->stream));

    int num_candidates = std::min(pImpl->cpu_valid_count[0], pImpl->MAX_OUTPUT_BBOX_COUNT);
    if (num_candidates <= 0) return;

    std::vector<float> cpu_buffer(num_candidates * 7);
    CHECK_CUDA(cudaMemcpyAsync(cpu_buffer.data(), pImpl->gpu_output_buffer, 
                               num_candidates * 7 * sizeof(float), cudaMemcpyDeviceToHost, pImpl->stream));
    CHECK_CUDA(cudaStreamSynchronize(pImpl->stream));

    output.clear();
    
    float r_w = pImpl->input_w / (float)pImpl->last_img_w;
    float r_h = pImpl->input_h / (float)pImpl->last_img_h;
    float scale = (r_w < r_h) ? r_w : r_h; // Min scale
    float offset_x = (pImpl->input_w - pImpl->last_img_w * scale) / 2.0f;
    float offset_y = (pImpl->input_h - pImpl->last_img_h * scale) / 2.0f;

    for (int i = 0; i < num_candidates; ++i) {
        float* item = &cpu_buffer[i * 7];
        
        if (item[6] != 1.0f) continue; 

        float left   = item[0]; 
        float top    = item[1]; 
        float right  = item[2]; 
        float bottom = item[3];
        float conf   = item[4]; 
        int class_id = (int)item[5];

        float x1 = (left - offset_x) / scale;
        float y1 = (top - offset_y) / scale;
        float x2 = (right - offset_x) / scale;
        float y2 = (bottom - offset_y) / scale;

        int x = std::max(0, std::min((int)x1, pImpl->last_img_w - 1));
        int y = std::max(0, std::min((int)y1, pImpl->last_img_h - 1));
        int w = std::min((int)(x2 - x1), pImpl->last_img_w - x);
        int h = std::min((int)(y2 - y1), pImpl->last_img_h - y);

        if (w > 0 && h > 0) {
            YoloDetection res;
            res.class_id = class_id;
            res.conf = conf;
            res.bbox = cv::Rect(x, y, w, h);
            output.push_back(res);
        }
    }
}

void YOLOv11::draw(cv::Mat& image, const std::vector<YoloDetection>& output)
{
    for (const auto& det : output) {
        cv::Scalar color = pImpl->class_colors[det.class_id % pImpl->class_colors.size()];
        cv::rectangle(image, det.bbox, color, 2);

        std::string name = (det.class_id >= 0 && det.class_id < CLASS_NAMES.size()) 
                           ? CLASS_NAMES[det.class_id] 
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