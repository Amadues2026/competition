/**
 * @file visualizer.cpp
 * @brief 异步可视化模块实现
 */

#include "visualizer.h"

#include <algorithm>
#include <iostream>
#include <utility>

Visualizer::Visualizer(std::chrono::milliseconds refresh_interval, const std::string& window_name)
    : refresh_interval_(std::max(refresh_interval, std::chrono::milliseconds(1))),
      window_name_(window_name),
      worker_(&Visualizer::run, this) {}

Visualizer::~Visualizer() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    stop_cv_.notify_one();
    if (worker_.joinable()) worker_.join();
}

void Visualizer::update(const cv::Mat& frame, std::vector<DetectedArmor> armors) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_frame_ = frame;
    latest_armors_ = std::move(armors);
}

void Visualizer::run() {
    bool window_created = false;
    while (true) {
        cv::Mat display;
        std::vector<DetectedArmor> armors;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_cv_.wait_for(lock, refresh_interval_, [this] { return stop_; })) break;
            if (latest_frame_.empty()) continue;

            display = latest_frame_.clone();
            armors = latest_armors_;
        }

        try {
            drawAndShow(display, armors);
            window_created = true;
        } catch (const cv::Exception& e) {
            std::cerr << "[Visualizer] GUI disabled after OpenCV error: " << e.what() << std::endl;
            break;
        }
    }

    if (window_created) {
        try {
            cv::destroyWindow(window_name_);
        } catch (const cv::Exception&) {
        }
    }
}

void Visualizer::drawAndShow(cv::Mat& display, const std::vector<DetectedArmor>& armors) {
    for (const auto& armor : armors) {
        cv::rectangle(display, armor.armorBbox, cv::Scalar(0, 255, 0), 2);
        cv::circle(display, armor.center, 5, cv::Scalar(0, 0, 255), -1);

        std::string label = "C" + std::to_string(armor.class_id) + ":" + std::to_string(armor.score).substr(0, 4);
        int info_x = armor.armorBbox.x + armor.armorBbox.width + 5;
        cv::putText(display, label, cv::Point(info_x, armor.armorBbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);

        if (armor.distance > 0.01f) {
            char distance_str[32];
            snprintf(distance_str, sizeof(distance_str), "%.2fm", armor.distance);
            cv::putText(display, distance_str, cv::Point(info_x, armor.armorBbox.y + armor.armorBbox.height + 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
        }
    }
    cv::imshow(window_name_, display);
    cv::waitKey(1);
}
