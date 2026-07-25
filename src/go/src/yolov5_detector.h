/**
 * @file yolov5_detector.h
 * @brief YOLOv5 TensorRT 目标检测器声明 —— 推理、PnP 测距、弹道补偿
 *
 * YOLOv5Detector 是视觉感知的核心模块，负责：
 *   1. TensorRT 引擎初始化和推理（GPU 加速）
 *   2. 目标检测（识别敌方装甲板）
 *   3. PnP 单目测距（从2D检测框解算3D距离和角度）
 *   4. 弹道补偿计算（考虑弹丸飞行时间+重力下坠）
 *
 * 推理流程：
 *   updateFrame()（相机回调） → detectArmors() → solvePnPYawPitchDist() → calculateBallisticCompensation()
 *
 * 可视化由 Visualizer 模块独立处理，通过 YAML 配置 enable_visualization 控制。
 *
 * PnP 测距原理：
 *   已知装甲板真实尺寸（宽/高）+ 相机内参（fx/fy/cx/cy）+ 2D检测框坐标
 *   → solvePnP 解算出相机坐标系下的 3D 位置 → 转换为 yaw/pitch/distance
 *
 * 弹道补偿：
 *   考虑弹丸飞行时间内的目标移动和重力下坠，
 *   输出"应该瞄准哪里"（而非目标当前位置）
 *
 * 距离滤波：
 *   使用 RollingMedianFilter 平滑 PnP 测距结果，消除异常跳变
 */

#ifndef COMPETITION_YOLOV5_DETECTOR_H
#define COMPETITION_YOLOV5_DETECTOR_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <mutex>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detection_types.h"
#include "logger.h"
#include "rolling_median_filter.h"

struct YOLOv5DetectorConfig {
    double conf_threshold = 0.1;
    double nms_iou_threshold = 0.45;
    double camera_fx = 554.26;
    double camera_fy = 554.26;
    double camera_cx = 320.0;
    double camera_cy = 240.0;
    double bullet_speed = 23.0;
    double gravity = 5.0;
    double armor_width = 1.0;
    double armor_height = 0.625;
    int distance_filter_window = 150;
};

class YOLOv5Detector {
private:
    int player_id_;         ///< 玩家ID（决定目标类别范围）
    cv::Mat latest_frame_;  ///< 最新相机帧（由 GoNode 回调更新）

    // ── TensorRT 引擎组件 ──
    std::unique_ptr<nvinfer1::IRuntime> runtime_;           ///< TensorRT 运行时
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;         ///< CUDA 推理引擎
    std::unique_ptr<nvinfer1::IExecutionContext> context_;  ///< 推理执行上下文

    // ── GPU/CPU 缓冲区 ──
    void* gpu_buffers_[2];           ///< GPU 输入/输出缓冲区
    cudaStream_t infer_stream_;      ///< CUDA 推理流
    std::vector<float> input_data_;  ///< 输入数据缓冲区（预分配）
    std::vector<float> cpu_output_;  ///< CPU 端输出缓冲区
    int output_dim1_;                ///< 输出维度1（检测框数量）
    int output_dim2_;                ///< 输出维度2（每框属性数）
    int input_h_, input_w_;          ///< 输入图像尺寸

    // ── 检测参数 ──
    int target_class_start_, target_class_end_;  ///< 目标类别ID范围
    float conf_threshold_;                       ///< 置信度阈值
    double nms_iou_threshold_;                   ///< NMS IoU 阈值
    Logger logger_;                              ///< TensorRT 日志回调

    // ── 模型和相机参数 ──
    std::string engine_path_;                               ///< TensorRT 引擎文件路径
    double camera_fx_, camera_fy_, camera_cx_, camera_cy_;  ///< 相机内参
    double bullet_speed_, gravity_;                         ///< 弹道参数
    double armor_width_, armor_height_;                     ///< 装甲板尺寸（米）

    // ── 预计算 PnP 常量（从配置加载后构造，避免重复分配） ──
    std::vector<cv::Point3f> corners_3d_;  ///< 装甲板 3D 角点（预计算）
    cv::Mat camera_matrix_;                ///< 相机内参矩阵（预计算）
    cv::Mat dist_coeffs_;                  ///< 畸变系数（模拟器无畸变）

    // ── 内部状态 ──
    mutable std::mutex mutex_;             ///< 互斥锁（保护 latest_frame_ + cached_armors_）
    RollingMedianFilter distance_filter_;  ///< 距离中值滤波器
    int infer_err_count_ = 0;              ///< 推理错误计数

    /**
     * @brief 单个检测框的内部表示
     */
    struct Detection {
        float x, y, w, h;          ///< 边界框（像素坐标）
        float score;               ///< 置信度
        int class_id;              ///< 类别ID
        float center_x, center_y;  ///< 中心点坐标
    };

    // ── 内部方法 ──
    void initTensorRT();                                     ///< 初始化 TensorRT 引擎
    cv::Mat preprocess(const cv::Mat& frame);                ///< 图像预处理（resize + 归一化）
    std::vector<Detection> infer(const cv::Mat& processed);  ///< TensorRT 推理
    std::vector<Detection> decodeDetections();               ///< 解码原始输出为检测框
    void cleanup();                                          ///< 释放 GPU 资源

public:
    /**
     * @brief 安全获取缓存的检测结果（线程安全，返回副本）
     * @return 检测到的装甲板列表副本
     */
    std::vector<DetectedArmor> getCachedArmors() const;

    YOLOv5Detector(int player_id, const std::string& engine_path, const YOLOv5DetectorConfig& config);

    /** @brief 析构函数（释放 GPU 资源） */
    ~YOLOv5Detector();

    /**
     * @brief 更新最新相机帧（由 GoNode::cameraCallback 调用）
     * 共享不可变 cv::Mat 缓冲区，不复制像素；可视化由 Visualizer 模块处理。
     */
    void updateFrame(const cv::Mat& frame);

    /** @brief 缓存检测结果（供外部模块读取和可视化） */
    void cacheDetections(const std::vector<DetectedArmor>& armors);

    /**
     * @brief 执行完整的检测流程（推理 + 后处理）
     * @return 检测到的装甲板列表
     */
    std::vector<DetectedArmor> detectArmors();

    // ── 弹道补偿计算 ──

    /**
     * @brief 计算弹道补偿后的云台指向角
     * @param yaw_offset    目标相对 yaw 偏移
     * @param pitch_offset  目标相对 pitch 偏移
     * @param distance      目标距离
     * @param current_yaw   当前云台 yaw
     * @param current_pitch 当前云台 pitch
     * @return 补偿后的 {yaw, pitch}
     */
    std::tuple<double, double> calculateBallisticCompensation(double yaw_offset, double pitch_offset, double distance,
                                                              double current_yaw, double current_pitch);

    /** @brief 原始 PnP 测距（仅返回距离） */
    double rawPnPDistance(const cv::Rect& armor_bbox);

    /**
     * @brief PnP 解算（返回距离 + 3D 位置）
     * @return {distance, Point3f(x, y, z)} 相机坐标系下的3D位置
     */
    std::tuple<double, cv::Point3f> solvePnPDistance(const cv::Rect& armor_bbox);

    /**
     * @brief PnP 完整解算（返回 yaw/pitch/distance）
     * @return {yaw_deg, pitch_deg, distance} 相机坐标系下的角度和距离
     */
    std::tuple<double, double, double> solvePnPYawPitchDist(const cv::Rect& armor_bbox);

private:
    std::vector<DetectedArmor>
        cached_armors_;  ///< 缓存的检测结果（getCachedArmors 加锁读取，cacheDetections 加锁写入）
};

#endif
