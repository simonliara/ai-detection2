#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace nvinfer1 { class ILogger; }
class Track;

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
    
    void draw(cv::Mat& image, const std::vector<YoloDetection>& output);
    void draw(cv::Mat& image, const std::vector<std::shared_ptr<Track>>& tracks);

    float getConfThreshold() const;
    float getNMSThreshold() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};