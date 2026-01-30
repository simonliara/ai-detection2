#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <dds/dds.hpp>

#include "DataBus/SensorData/Image.hpp"
#include "DataBus/Identity.hpp"
#include "DataBus/Time.hpp"

#include <data_bus/topics/sensor_data.h>   
#include <data_bus/qos.h>                 

using namespace std::chrono;

static constexpr int    kDomainId   = 0;
static constexpr int    kCamIndex   = 0;
static constexpr double kFpsDefault = 30.0;
static const std::string kReferenceFrame = "video_source";

static constexpr bool kUseJpeg      = false;
static constexpr int  kJpegQuality  = 80;

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

static DataBus::Time makeMonotonicTimeNow() {
    const auto ns = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();

    DataBus::Time t{};
    const int64_t sec  = ns / 1000000000LL;
    const int64_t nsec = ns % 1000000000LL;
    t.seconds(sec);
    t.nanoSeconds((int32_t)nsec);
    return t;
}

static void fillBGR24Image(DataBus::SensorData::Image& out, const cv::Mat& bgr) {
    out.width((uint32_t)bgr.cols);
    out.height((uint32_t)bgr.rows);
    out.encoding(DataBus::SensorData::Encoding::BGR24);

    const size_t bytes = (size_t)bgr.total() * (size_t)bgr.elemSize(); // H*W*3
    std::vector<uint8_t> buf(bytes);
    std::memcpy(buf.data(), bgr.data, bytes);
    out.data(std::move(buf));
}

static bool fillJpegImage(DataBus::SensorData::Image& out, const cv::Mat& bgr, int jpegQuality) {
    std::vector<uint8_t> jpg;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, jpegQuality };
    if (!cv::imencode(".jpg", bgr, jpg, params)) return false;

    out.width((uint32_t)bgr.cols);
    out.height((uint32_t)bgr.rows);
    out.encoding(DataBus::SensorData::Encoding::JPEG);
    out.data(std::move(jpg));
    return true;
}

static void setCycloneEnvOnce() {
#if defined(_WIN32)
    _putenv_s("CYCLONEDDS_URI", kCycloneDDSUri);
#else
    ::setenv("CYCLONEDDS_URI", kCycloneDDSUri, /*overwrite=*/1);
#endif
}

int main(int argc, char** argv) {
    setCycloneEnvOnce();

    // 1. Determine Source
    cv::VideoCapture cap;
    std::string sourceName;
    double targetFps = kFpsDefault;

    if (argc > 1) {
        // Use video file path
        sourceName = argv[1];
        cap.open(sourceName);
        
        // Try to get FPS from the video file itself
        double videoFps = cap.get(cv::CAP_PROP_FPS);
        if (videoFps > 0) targetFps = videoFps;
    } else {
        // Use default webcam
        sourceName = "Webcam " + std::to_string(kCamIndex);
        cap.open(kCamIndex);
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open source: " << sourceName << "\n";
        return 1;
    }

    dds::domain::DomainParticipant participant(kDomainId);
    auto writer = DataBus::Topics::SensorData::getImagesWriter(participant);

    std::cout
        << "Publishing DataBus::SensorData::TimedImage\n"
        << "  Topic:  /data_bus/sensor_data/images\n"
        << "  Source: " << sourceName << "\n"
        << "  FPS:    " << targetFps << "\n"
        << "  Enc:    " << (kUseJpeg ? "JPEG" : "BGR24") << "\n";

    const auto framePeriod = duration<double>(1.0 / targetFps);

    while (true) {
        const auto loopStart = steady_clock::now();

        cv::Mat frame;
        cap >> frame;

        // 2. Handle end of video or empty frames
        if (frame.empty()) {
            if (argc > 1) {
                std::cout << "End of video file reached.\n";
                break; 
            }
            continue;
        }

        // --- Frame Processing (BGR Conversion) ---
        if (frame.type() == CV_8UC1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        } else if (frame.type() != CV_8UC3) {
            cv::Mat tmp;
            frame.convertTo(tmp, CV_8U);
            if (tmp.channels() == 1) cv::cvtColor(tmp, frame, cv::COLOR_GRAY2BGR);
            else frame = tmp;
        }

        // --- DataBus Message Construction ---
        DataBus::SensorData::TimedImage msg{};
        msg.referenceFrame(kReferenceFrame);
        msg.imageTime(makeMonotonicTimeNow());

        DataBus::Identity id{}; 
        msg.sourceIdentity(id);

        DataBus::SensorData::Image img{};
        if (kUseJpeg) {
            if (!fillJpegImage(img, frame, kJpegQuality)) continue;
        } else {
            fillBGR24Image(img, frame);
        }
        msg.image(std::move(img));

        writer.write(msg);

        // 3. UI and Timing
        cv::imshow("Publisher Preview", frame);
        if (cv::waitKey(1) == 27) break;

        const auto elapsed = steady_clock::now() - loopStart;
        if (elapsed < framePeriod) {
            std::this_thread::sleep_for(duration_cast<nanoseconds>(framePeriod - elapsed));
        }
    }

    cv::destroyAllWindows();
    cap.release();
    return 0;
}