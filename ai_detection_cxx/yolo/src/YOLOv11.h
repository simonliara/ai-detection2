#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include "ByteTrack/STrack.h"
#include "ByteTrack/Object.h"

namespace nvinfer1 { class ILogger; }
class Track;
namespace byte_track {
    class STrack; 
    class Object;
}

struct YoloStats {
    float pre;
    float infer;
    float post;
};

struct YoloDetection
{
    float conf;
    int class_id;
    cv::Rect bbox;
};

class YOLOv11
{
public:
    YOLOv11(std::string model_path, nvinfer1::ILogger& logger, float conf_thresh, float nms_thresh);
    ~YOLOv11();

    void preprocess(cv::Mat& image);
    void infer();
    void postprocess(std::vector<YoloDetection>& output);
    YoloStats getStats() const;
    
    void draw(cv::Mat& image, const std::vector<YoloDetection>& output);
    void draw(cv::Mat& image, const std::vector<std::shared_ptr<Track>>& tracks);
    void draw(cv::Mat& image, const std::vector<std::shared_ptr<byte_track::STrack>>& tracks);

    float getConfThreshold() const;
    float getNMSThreshold() const;
    static std::vector<byte_track::Object> toByteTrackObjects(
        const std::vector<YoloDetection>& yolo, 
        float conf_thresh = 0.25f
    );
    std::vector<std::string> getClassNames() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};