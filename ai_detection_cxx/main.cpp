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

static constexpr int kDomainId = 0;

static const char* kCycloneDDSUri =
    "<CycloneDDS>"
      "<Domain>"
        "<General>"
          "<Interfaces>"
            "<NetworkInterface address=\"127.0.0.1\"/>"
          "</Interfaces>"
          "<AllowMulticast>true</AllowMulticast>"
          "<EnableMulticastLoopback>true</EnableMulticastLoopback>"
        "</General>"
      "</Domain>"
    "</CycloneDDS>";

static const std::string kYoloEngine = "ai_detection_cxx/yolo/weights/yolo11s_static.fp16.engine";
static constexpr float kConfThresh = 0.1f;
static constexpr float kNmsThresh  = 0.7f;

static const std::string kTrackerIni = "ai_detection_cxx/botsort/config/tracker.ini";
static const std::string kGmcIni     = "ai_detection_cxx/botsort/config/gmc.ini";
static const std::string kReidIni    = "ai_detection_cxx/botsort/config/reid.ini";
static const std::string kReidOnnx   = "ai_detection_cxx/botsort/weights/osnet_x0_25_market1501.onnx"; 

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

static inline std::vector<Detection> toBotSortDetections(
    const std::vector<YoloDetection>& yolo,
    int imgW,
    int imgH,
    float conf_thresh
) {
    std::vector<Detection> out;
    out.reserve(yolo.size());
    for (const auto& d : yolo) {
        if (d.conf < conf_thresh) continue;

        cv::Rect box = clampRect(d.bbox, imgW, imgH);
        if (box.area() <= 0) continue;

        Detection bd;
        bd.class_id   = d.class_id;
        bd.confidence = d.conf;
        bd.bbox_tlwh  = cv::Rect_<float>((float)box.x, (float)box.y, (float)box.width, (float)box.height);
        out.push_back(bd);
    }
    return out;
}

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
} g_trtLogger;

static void setCycloneEnvOnce() {
#if defined(_WIN32)
    _putenv_s("CYCLONEDDS_URI", kCycloneDDSUri);
#else
    ::setenv("CYCLONEDDS_URI", kCycloneDDSUri, 1);
#endif
}

class DataBusDetectorNode {
public:
    DataBusDetectorNode()
    : participant_(kDomainId),
      imageReader_(DataBus::Topics::SensorData::getImagesReader(participant_)),
      detWriter_(DataBus::Topics::Perception::getObjectDetectionWriter(participant_)),
      model_(kYoloEngine, g_trtLogger, kConfThresh, kNmsThresh),
      botsort_(std::make_unique<BoTSORT>(kTrackerIni, kGmcIni, kReidIni, kReidOnnx)),
      tracker_(std::make_unique<byte_track::BYTETracker>())
    {}

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
            } 
            else {
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

        auto inputs = YOLOv11::toByteTrackObjects(yoloDetections, kConfThresh);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto tracks = tracker_->update(inputs); 
        auto t3 = std::chrono::high_resolution_clock::now();

        for (const auto& tp : tracks) {
            if (!tp) continue;
            const int trackId = static_cast<int>(tp->getTrackId());
            const auto& rect  = tp->getRect(); 

            cv::Rect boxPx = clampRect(cv::Rect(rect.x(), rect.y(), rect.width(), rect.height()), bgr.cols, bgr.rows);
            if (boxPx.area() <= 0) continue;

            int classId = tp->getClassId();
            std::string className = (classId < model_.getClassNames().size()) ? model_.getClassNames()[classId] : "unknown";
            
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
        spdlog::info("Inference: {:.2f}ms | Track: {:.2f}ms | Total: {:.2f}ms", yolo_ms, track_ms, 
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
            return cv::Mat(H, W, CV_8UC3, const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(data.data())));
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
    dds::domain::DomainParticipant participant_;
    DataBus::Topics::SensorData::ImagesReader imageReader_;
    DataBus::Topics::Perception::ObjectDetectionWriter detWriter_;

    YOLOv11 model_;
    std::unique_ptr<BoTSORT> botsort_;
    bool use_botsort_ = true;
    std::unique_ptr<byte_track::BYTETracker> tracker_;
    std::unordered_map<int, std::string> trackClass_;
    cv::Mat jpegScratchBuffer_;
    static constexpr bool kShowUi = true;
    static constexpr const char* kWinName = "yolo+tracking";
};

int main() {
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    setCycloneEnvOnce();

    spdlog::info("Starting DataBus detector (domain={}, iface=127.0.0.1)", kDomainId);
    spdlog::info("YOLO engine: {}", kYoloEngine);
    spdlog::info("BoTSORT cfg: {}, {}, {}", kTrackerIni, kGmcIni, kReidIni);

    DataBusDetectorNode node;
    node.loop();

    spdlog::info("Exiting");
    return 0;
}
