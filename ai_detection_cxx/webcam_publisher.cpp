#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>
#include <dds/dds.hpp>

#include "DataBus/SensorData/Image.hpp"
#include "DataBus/Identity.hpp"
#include "DataBus/Time.hpp"

#include <data_bus/topics/sensor_data.h>
#include <data_bus/qos.h>

using namespace std::chrono;

struct AppConfig {
    bool verbose = false;
    int domainId = 0;
    std::string iface = "127.0.0.1";
    bool allowMulticast = true;
    bool enableMulticastLoopback = true;

    int camIndex = 0;
    std::string videoFile = "";
    double fps = 30.0;

    std::string referenceFrame = "video_source";
    std::string sourceNameOverride = "";
    bool useJpeg = false;
    int jpegQuality = 80;
};

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

static void normalizeToBgr(cv::Mat& frame) {
    if (frame.empty()) return;

    if (frame.type() == CV_8UC1) {
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        return;
    }

    if (frame.type() == CV_8UC3) return;

    cv::Mat tmp;
    frame.convertTo(tmp, CV_8U);

    if (tmp.channels() == 1) {
        cv::cvtColor(tmp, frame, cv::COLOR_GRAY2BGR);
    } else if (tmp.channels() == 3) {
        frame = tmp;
    } else if (tmp.channels() == 4) {
        cv::cvtColor(tmp, frame, cv::COLOR_BGRA2BGR);
    } else {
        std::vector<cv::Mat> ch;
        cv::split(tmp, ch);
        if (!ch.empty()) {
            cv::Mat gray = ch[0];
            cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
        } else {
            frame.release();
        }
    }
}

int main(int argc, char** argv) {
    cxxopts::Options options("DataBus webcam/video publisher");

    // clang-format off
    options.add_options()
        ("h,help", "Show help")
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))

        ("domain-id", "DataBus domain id", cxxopts::value<int>())
        ("iface", "CycloneDDS interface address", cxxopts::value<std::string>()->default_value("127.0.0.1"))
        ("allow-multicast", "CycloneDDS AllowMulticast", cxxopts::value<bool>()->default_value("true"))
        ("enable-multicast-loopback", "CycloneDDS EnableMulticastLoopback", cxxopts::value<bool>()->default_value("true"))

        ("cam-index", "OpenCV camera index", cxxopts::value<int>()->default_value("0"))
        ("video", "Path/URL to video source (if set, overrides cam-index)", cxxopts::value<std::string>()->default_value(""))
        ("fps", "Target FPS (<=0 means: if video file, use file FPS; else 30)", cxxopts::value<double>()->default_value("30.0"))

        ("reference-frame", "TimedImage referenceFrame", cxxopts::value<std::string>()->default_value("video_source"))
        ("source-name", "Override source name used in logs (optional)", cxxopts::value<std::string>()->default_value(""))

        ("use-jpeg", "Publish JPEG instead of BGR24", cxxopts::value<bool>()->default_value("false"))
        ("jpeg-quality", "JPEG quality (1-100)", cxxopts::value<int>()->default_value("80"));
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

    cfg.camIndex = args["cam-index"].as<int>();
    cfg.videoFile = args["video"].as<std::string>();
    cfg.fps = args["fps"].as<double>();

    cfg.referenceFrame = args["reference-frame"].as<std::string>();
    cfg.sourceNameOverride = args["source-name"].as<std::string>();

    cfg.useJpeg = args["use-jpeg"].as<bool>();
    cfg.jpegQuality = args["jpeg-quality"].as<int>();
    if (cfg.jpegQuality < 1) cfg.jpegQuality = 1;
    if (cfg.jpegQuality > 100) cfg.jpegQuality = 100;

    spdlog::set_level(spdlog::level::info);
    if (cfg.verbose) spdlog::set_level(spdlog::level::debug);

    setCycloneEnvOnce(cfg.iface, cfg.allowMulticast, cfg.enableMulticastLoopback);

    cv::VideoCapture cap;
    std::string sourceName;

    const bool useVideo = !cfg.videoFile.empty();
    if (useVideo) {
        cap.open(cfg.videoFile);
        sourceName = cfg.videoFile;

        if (cfg.fps <= 0.0) {
            double videoFps = cap.get(cv::CAP_PROP_FPS);
            cfg.fps = (videoFps > 0.0) ? videoFps : 30.0;
        }
    } else {
        cap.open(cfg.camIndex);
        sourceName = "Webcam " + std::to_string(cfg.camIndex);

        if (cfg.fps <= 0.0) cfg.fps = 30.0;
    }

    if (!cfg.sourceNameOverride.empty()) {
        sourceName = cfg.sourceNameOverride;
    }

    if (!cap.isOpened()) {
        spdlog::error("Failed to open source: {}", sourceName);
        return 1;
    }

    dds::domain::DomainParticipant participant(cfg.domainId);
    auto writer = DataBus::Topics::SensorData::getImagesWriter(participant);

    spdlog::info("Publishing DataBus::SensorData::TimedImage");
    spdlog::info("  Topic:  /data_bus/sensor_data/images");
    spdlog::info("  Source: {}", sourceName);
    spdlog::info("  FPS:    {}", cfg.fps);
    spdlog::info("  Enc:    {}", (cfg.useJpeg ? "JPEG" : "BGR24"));
    if (cfg.useJpeg) spdlog::info("  JPEG Q: {}", cfg.jpegQuality);
    spdlog::info("  Frame:  {}", cfg.referenceFrame);
    spdlog::info("  DDS:    domain={} iface={} allowMulticast={} loopback={}",
                 cfg.domainId, cfg.iface, cfg.allowMulticast, cfg.enableMulticastLoopback);

    const auto framePeriod = duration<double>(1.0 / cfg.fps);

    while (true) {
        const auto loopStart = steady_clock::now();

        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            if (useVideo) {
                spdlog::info("End of video source reached.");
                break;
            }
            continue;
        }

        normalizeToBgr(frame);
        if (frame.empty()) continue;

        DataBus::SensorData::TimedImage msg{};
        msg.referenceFrame(cfg.referenceFrame);
        msg.imageTime(makeMonotonicTimeNow());

        DataBus::Identity id{};
        msg.sourceIdentity(id);

        DataBus::SensorData::Image img{};
        if (cfg.useJpeg) {
            if (!fillJpegImage(img, frame, cfg.jpegQuality)) continue;
        } else {
            fillBGR24Image(img, frame);
        }
        msg.image(std::move(img));

        writer.write(msg);

        spdlog::info("Published frame: {}x{}", msg.image().width(), msg.image().height());

        const auto elapsed = steady_clock::now() - loopStart;
        if (elapsed < framePeriod) {
            std::this_thread::sleep_for(duration_cast<nanoseconds>(framePeriod - elapsed));
        }
    }

    cv::destroyAllWindows();
    cap.release();
    return 0;
}
