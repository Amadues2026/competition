/**
 * @file visualizer.h
 * @brief 异步可视化模块 —— 在独立线程中绘制和显示检测结果
 *
 * cameraCallback 只调用 update() 发布最新帧，不执行 cv::imshow/cv::waitKey。
 * 工作线程按 YAML 配置的刷新间隔读取最新快照，因此 GUI 不会阻塞 ROS 回调。
 * 输入帧按 cv::Mat 引用计数共享，只有实际绘制时才克隆一次像素缓冲区。
 */

#ifndef COMPETITION_VISUALIZER_H
#define COMPETITION_VISUALIZER_H

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detection_types.h"

class Visualizer {
public:
    explicit Visualizer(std::chrono::milliseconds refresh_interval,
                        const std::string& window_name = "YOLOv5n Detection");
    ~Visualizer();

    Visualizer(const Visualizer&) = delete;
    Visualizer& operator=(const Visualizer&) = delete;

    /** @brief 发布最新不可变帧和检测结果；仅复制 cv::Mat 头及小型结果数组 */
    void update(const cv::Mat& frame, std::vector<DetectedArmor> armors);

private:
    void run();
    void drawAndShow(cv::Mat& display, const std::vector<DetectedArmor>& armors);

    std::chrono::milliseconds refresh_interval_;
    std::string window_name_;
    std::mutex mutex_;
    std::condition_variable stop_cv_;
    cv::Mat latest_frame_;
    std::vector<DetectedArmor> latest_armors_;
    bool stop_ = false;
    std::thread worker_;
};

#endif
