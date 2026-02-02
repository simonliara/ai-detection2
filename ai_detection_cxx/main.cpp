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

static const std::string kYoloEngine = "yolo/weights/yolo11s_static.fp16.engine";
static constexpr float kConfThresh = 0.25f;
static constexpr float kNmsThresh  = 0.45f;

static const std::string kTrackerIni = "botsort/config/tracker.ini";
static const std::string kGmcIni     = "botsort/config/gmc.ini";
static const std::string kReidIni    = "botsort/config/reid.ini";
static const std::string kReidOnnx   = "botsort/weights/osnet_x0_25_market1501.onnx"; 

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

static inline std::string classIdToName(int cid) {
    if (cid == 0) return "person";
    if (cid == 2 || cid == 7) return "vehicle";
    return "unknown";
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

static std::optional<cv::Mat> decodeTimedImage(const DataBus::SensorData::TimedImage& msg) {
    const auto& im = msg.image();

    if (im.encoding() == DataBus::SensorData::Encoding::BGR24) {
        const int H = (int)im.height();
        const int W = (int)im.width();
        const auto& data = im.data();
        if ((int)data.size() < H * W * 3) return std::nullopt;

        cv::Mat view(H, W, CV_8UC3, (void*)data.data());
        return view.clone();
    }

    if (im.encoding() == DataBus::SensorData::Encoding::JPEG) {
        const auto& data = im.data();
        cv::Mat buf(1, (int)data.size(), CV_8UC1, (void*)data.data());
        cv::Mat decoded = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (decoded.empty()) return std::nullopt;
        return decoded;
    }

    return std::nullopt;
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
      botsort_(std::make_unique<BoTSORT>(kTrackerIni, kGmcIni, kReidIni, kReidOnnx))
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
            auto t2 = std::chrono::high_resolution_clock::now();
            double loop_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            spdlog::info("Loop time: {} ms", loop_ms);
        }
    }

private:
    void processFrame(const cv::Mat& bgr, const DataBus::SensorData::TimedImage& timedImage) {
        auto ta = std::chrono::high_resolution_clock::now();
        if (bgr.empty()) return;

        cv::Mat image = bgr.clone();

        std::vector<YoloDetection> yoloDetections;

        auto t0 = std::chrono::high_resolution_clock::now();
        model_.preprocess(image);
        model_.infer();
        model_.postprocess(yoloDetections);
        auto t1 = std::chrono::high_resolution_clock::now();

        auto botDets = toBotSortDetections(yoloDetections, image.cols, image.rows, kConfThresh);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto tracks = botsort_->track(botDets, image);
        auto t3 = std::chrono::high_resolution_clock::now();

        for (const auto& tp : tracks) {
            if (!tp) continue;

            const int trackId = tp->track_id;

            const auto tlwh = tp->get_tlwh();
            if (tlwh.size() != 4) continue;

            cv::Rect boxPx((int)tlwh[0], (int)tlwh[1], (int)tlwh[2], (int)tlwh[3]);
            boxPx = clampRect(boxPx, image.cols, image.rows);
            if (boxPx.area() <= 0) continue;

            const float conf = tp->get_score();
            const int classId = (int)tp->get_class_id();

            std::string className = classIdToName(classId);

            auto it = trackClass_.find(trackId);
            if (it == trackClass_.end()) {
                if (className == "unknown") continue;
                trackClass_[trackId] = className;
            } else {
                className = it->second;
            }

            double cx, cy, sx, sy;
            rectToXywhn(boxPx, image.cols, image.rows, cx, cy, sx, sy);

            DataBus::Perception::ObjectDetection msg;
            msg.id((uint64_t)trackId);
            msg.sourceIdentity(timedImage.sourceIdentity());
            msg.time(timedImage.imageTime());
            msg.classification(className);
            msg.confidenceScore(conf);
            msg.referenceFrame(timedImage.referenceFrame());
            msg.centerX(cx);
            msg.centerY(cy);
            msg.sizeX(sx);
            msg.sizeY(sy);

            detWriter_.write(msg);
        }

        model_.draw(image, tracks);

        auto tb = std::chrono::high_resolution_clock::now();
        double yolo_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double track_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double tot_ms = std::chrono::duration<double, std::milli>(tb - ta).count();

        std::printf("YOLO: %6.2f ms | Track: %6.2f ms | dets=%zu tracks=%zu | tot=%6.2f\n",
                    yolo_ms, track_ms, botDets.size(), tracks.size(), tot_ms);

        // if (kShowUi) {
        //     cv::imshow(kWinName, image);
        //     if (cv::waitKey(1) == 27) {
        //         g_running = false;
        //     }
        // }
    }

private:
    dds::domain::DomainParticipant participant_;
    DataBus::Topics::SensorData::ImagesReader imageReader_;
    DataBus::Topics::Perception::ObjectDetectionWriter detWriter_;

    YOLOv11 model_;
    std::unique_ptr<BoTSORT> botsort_;
    std::unordered_map<int, std::string> trackClass_;
    static constexpr bool kShowUi = true;
    static constexpr const char* kWinName = "yolo+botsort";
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
