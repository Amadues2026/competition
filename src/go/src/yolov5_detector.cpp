/**
 * @file yolov5_detector.cpp
 * @brief YOLOv5 目标检测器 —— TensorRT 推理 + PnP 测距 + 弹道补偿
 *
 * 本模块封装了从图像输入到瞄准输出的完整感知链路：
 *
 *   相机图像 → 预处理 → TensorRT 推理 → NMS 后处理 → 装甲板检测
 *                                                        ↓
 *                                                  PnP 求解距离
 *                                                        ↓
 *                                              弹道补偿（重力下坠）
 *                                                        ↓
 *                                                 最终 yaw/pitch
 *
 * 模型信息：
 *   - 基础网络：YOLOv5n（轻量级）
 *   - 推理引擎：TensorRT FP16
 *   - 输入尺寸：224×416（高×宽，竖屏适配）
 *   - 类别数：2（0=红色装甲板，1=蓝色装甲板）
 *   - 部署路径：/home/robot/competition/models/yolov5n_d2c2_fp16.engine
 *
 * PnP 测距原理：
 *   已知装甲板的物理尺寸（1.0×0.625 地图单位）和相机内参，
 *   通过四个角点的 2D-3D 对应关系求解相机坐标系下的位姿（rvec, tvec），
 *   tvec 的 z 分量即为距离。三个 PnP 方法的区别：
 *   - rawPnPDistance：快速测距，无滤波，用于距离门控初筛
 *   - solvePnPDistance：带中值滤波，用于确定最终跟踪目标的距离
 *   - solvePnPYawPitchDist：输出 yaw/pitch/distance，送入 EKF 滤波
 *
 * 弹道补偿：
 *   子弹飞行过程中受重力影响下坠，需要额外抬高 pitch 角。
 *   补偿公式：pitch_comp = asin(0.5 × g × t² / d)，其中 t = d / v_bullet
 *
 * 线程安全：
 *   - latest_frame_ 和 cached_armors_ 通过 mutex_ 保护
 *   - TensorRT 推理在独立线程中异步执行（通过 CUDA stream）
 */

#include "yolov5_detector.h"

#include <algorithm>
#include <fstream>
#include <iostream>

/**
 * @brief 构造函数 —— 初始化参数 + 加载 TensorRT 引擎
 * @param player_id 玩家ID（1=蓝方，2=红方），决定检测哪个颜色的装甲板
 * @param engine_path TensorRT 引擎文件路径
 * @param config 已加载并验证的检测、相机和弹道参数
 *
 * 类别过滤逻辑：
 *   player_id=1（蓝方）→ 打红方 → target_class=0（红色）
 *   player_id=2（红方）→ 打蓝方 → target_class=1（蓝色）
 *   只保留目标类别的检测结果，忽略己方颜色
 */
YOLOv5Detector::YOLOv5Detector(int player_id, const std::string& engine_path, const YOLOv5DetectorConfig& config)
    : player_id_(player_id),
      conf_threshold_(static_cast<float>(config.conf_threshold)),
      nms_iou_threshold_(config.nms_iou_threshold),
      engine_path_(engine_path),
      camera_fx_(config.camera_fx),
      camera_fy_(config.camera_fy),
      camera_cx_(config.camera_cx),
      camera_cy_(config.camera_cy),
      bullet_speed_(config.bullet_speed),
      gravity_(config.gravity),
      armor_width_(config.armor_width),
      armor_height_(config.armor_height) {
    gpu_buffers_[0] = nullptr;
    gpu_buffers_[1] = nullptr;
    infer_stream_ = nullptr;

    // 蓝方打红(class0)，红方打蓝(class1)
    if (player_id == 1) {
        target_class_start_ = 0;
        target_class_end_ = 0;
    } else {
        target_class_start_ = 1;
        target_class_end_ = 1;
    }

    // 预计算 PnP 常量：3D 角点、相机矩阵、畸变系数
    corners_3d_ = {
        cv::Point3f(-armor_width_ / 2, -armor_height_ / 2, 0), cv::Point3f(armor_width_ / 2, -armor_height_ / 2, 0),
        cv::Point3f(-armor_width_ / 2, armor_height_ / 2, 0), cv::Point3f(armor_width_ / 2, armor_height_ / 2, 0)};
    camera_matrix_ = (cv::Mat_<double>(3, 3) << camera_fx_, 0, camera_cx_, 0, camera_fy_, camera_cy_, 0, 0, 1);
    dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64F);

    distance_filter_ = RollingMedianFilter(config.distance_filter_window);
    try {
        initTensorRT();
    } catch (...) {
        cleanup();
        throw;
    }

    // 预分配推理输入缓冲区（引擎加载后才知道尺寸）
    input_data_.resize(3 * input_h_ * input_w_);
}

YOLOv5Detector::~YOLOv5Detector() { cleanup(); }

/**
 * @brief 更新最新相机帧（由 GoNode::cameraCallback 调用）
 *
 * 只保存不可变帧的共享所有权，不复制像素，也不包含 GUI 绘制逻辑。
 * 调用方传入后不得修改该帧；cv::Mat 引用计数保证原调用对象销毁后数据仍有效。
 * 可视化由 Visualizer 模块单独处理，通过 YAML 配置 enable_visualization 控制。
 */
void YOLOv5Detector::updateFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_frame_ = frame;
}

/**
 * @brief 缓存检测结果，供外部模块读取和可视化使用
 * 由 CombatManager::shootLoop 在推理完成后调用
 */
void YOLOv5Detector::cacheDetections(const std::vector<DetectedArmor>& armors) {
    std::lock_guard<std::mutex> lock(mutex_);
    cached_armors_ = armors;
}

std::vector<DetectedArmor> YOLOv5Detector::getCachedArmors() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cached_armors_;
}

/**
 * @brief 完整检测流水线：快照帧 → 预处理 → 推理 → 坐标映射
 * @return 检测到的装甲板列表（坐标已映射回原始图像尺寸）
 *
 * 流程：
 *   1. 加锁复制 cv::Mat 头，共享不可变像素缓冲区
 *   2. preprocess：resize + BGR2RGB + 归一化
 *   3. infer：TensorRT 推理
 *   4. 将推理输出的归一化坐标映射回原始图像尺寸
 */
std::vector<DetectedArmor> YOLOv5Detector::detectArmors() {
    std::vector<DetectedArmor> detected_armors;
    int orig_h, orig_w;
    cv::Mat frame_snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latest_frame_.empty()) return detected_armors;
        frame_snapshot = latest_frame_;
    }
    orig_h = frame_snapshot.rows;
    orig_w = frame_snapshot.cols;

    cv::Mat processed = preprocess(frame_snapshot);
    auto detections = infer(processed);

    // 将模型输入尺寸坐标映射回原始图像坐标
    for (const auto& det : detections) {
        float mapped_x = det.x / input_w_ * orig_w;
        float mapped_y = det.y / input_h_ * orig_h;
        float mapped_w = det.w / input_w_ * orig_w;
        float mapped_h = det.h / input_h_ * orig_h;
        float mapped_cx = det.center_x / input_w_ * orig_w;
        float mapped_cy = det.center_y / input_h_ * orig_h;

        DetectedArmor armor;
        armor.center = cv::Point2f(mapped_cx, mapped_cy);
        armor.armorBbox = cv::Rect(mapped_x, mapped_y, mapped_w, mapped_h);
        armor.score = det.score;
        armor.class_id = det.class_id;
        detected_armors.push_back(armor);
    }

    return detected_armors;
}

/**
 * @brief 弹道补偿计算 —— 根据距离补偿重力下坠
 * @param yaw_offset 目标相对于当前云台的 yaw 偏移（度）
 * @param pitch_offset 目标相对于当前云台的 pitch 偏移（度）
 * @param distance 目标距离（地图单位）
 * @param current_yaw 当前云台 yaw（度）
 * @param current_pitch 当前云台 pitch（度）
 * @return 补偿后的绝对 yaw/pitch（度）
 *
 * 补偿原理：
 *   子弹飞行时间 t = distance / bullet_speed
 *   重力下坠量 h = 0.5 × g × t²
 *   需要抬高的角度 = asin(h / distance)
 *   drop_ratio 限制在 ±0.99 防止 asin 定义域溢出
 */
std::tuple<double, double> YOLOv5Detector::calculateBallisticCompensation(double yaw_offset, double pitch_offset,
                                                                          double distance, double current_yaw,
                                                                          double current_pitch) {
    if (distance < 1e-6) return {current_yaw + yaw_offset, current_pitch + pitch_offset};
    double t_flight = distance / bullet_speed_;
    double gravity_drop = 0.5 * gravity_ * t_flight * t_flight;
    double drop_ratio = gravity_drop / distance;
    if (drop_ratio > 0.99) drop_ratio = 0.99;
    if (drop_ratio < -0.99) drop_ratio = -0.99;
    double pitch_compensation = std::asin(drop_ratio) * 180.0 / CV_PI;
    return {current_yaw + yaw_offset, current_pitch + pitch_offset + pitch_compensation};
}

/**
 * @brief 快速 PnP 测距（无滤波）
 * @param armor_bbox 装甲板的 2D 检测框
 * @return 相机坐标系下的 z 轴距离，失败返回 999.0
 *
 * 用于 CombatManager 的距离门控初筛——只需要距离做比较，不需要平滑
 * 3D点假设装甲板是平面矩形，四个角点在 z=0 平面上
 */
double YOLOv5Detector::rawPnPDistance(const cv::Rect& armor_bbox) {
    if (armor_bbox.width <= 0 || armor_bbox.height <= 0) return 999.0;
    std::vector<cv::Point2f> corners_2d = {
        cv::Point2f(armor_bbox.x, armor_bbox.y), cv::Point2f(armor_bbox.x + armor_bbox.width, armor_bbox.y),
        cv::Point2f(armor_bbox.x, armor_bbox.y + armor_bbox.height),
        cv::Point2f(armor_bbox.x + armor_bbox.width, armor_bbox.y + armor_bbox.height)};
    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(corners_3d_, corners_2d, camera_matrix_, dist_coeffs_, rvec, tvec);
    if (success && !tvec.empty()) return tvec.at<double>(2);
    return 999.0;
}

/**
 * @brief PnP 测距（带中值滤波）
 * @param armor_bbox 装甲板的 2D 检测框
 * @return {滤波后的距离, 3D位置向量}
 *
 * 与 rawPnPDistance 的区别：
 *   - 距离经过 RollingMedianFilter（窗口150帧）平滑，消除跳变
 *   - 返回完整 3D 位置向量（用于距离门控后确定最终跟踪目标）
 *   - 失败时返回默认距离 3.0m 而非 999.0（避免 EKF 初始化异常）
 */
std::tuple<double, cv::Point3f> YOLOv5Detector::solvePnPDistance(const cv::Rect& armor_bbox) {
    if (armor_bbox.width <= 0 || armor_bbox.height <= 0) {
        return {3.0, cv::Point3f(0, 0, 3.0)};
    }
    std::vector<cv::Point2f> corners_2d = {
        cv::Point2f(armor_bbox.x, armor_bbox.y), cv::Point2f(armor_bbox.x + armor_bbox.width, armor_bbox.y),
        cv::Point2f(armor_bbox.x, armor_bbox.y + armor_bbox.height),
        cv::Point2f(armor_bbox.x + armor_bbox.width, armor_bbox.y + armor_bbox.height)};
    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(corners_3d_, corners_2d, camera_matrix_, dist_coeffs_, rvec, tvec);
    if (success && !tvec.empty()) {
        double distance = tvec.at<double>(2);
        distance_filter_.update(distance);  // 中值滤波平滑
        cv::Point3f position(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        return {distance_filter_.getFiltered(), position};
    }
    return {3.0, cv::Point3f(0, 0, 3.0)};
}

/**
 * @brief PnP 完整求解 —— 输出 yaw/pitch/distance（弧度）
 * @param armor_bbox 装甲板的 2D 检测框
 * @return {yaw(弧度), pitch(弧度), distance(地图单位)}
 *
 * 与前两个 PnP 方法的区别：
 *   - 输出的是角度（弧度），可直接送入 EKF 滤波
 *   - yaw = atan2(tx, tz)：水平偏角
 *   - pitch = atan2(-ty, √(tx²+tz²))：垂直偏角（负号因为相机坐标系y轴向下）
 *   - distance = √(tx²+ty²+tz²)：欧氏距离
 *   - 无滤波（EKF 在 CombatManager 中负责滤波）
 */
std::tuple<double, double, double> YOLOv5Detector::solvePnPYawPitchDist(const cv::Rect& armor_bbox) {
    if (armor_bbox.width <= 0 || armor_bbox.height <= 0) {
        return {0.0, 0.0, 3.0};
    }
    std::vector<cv::Point2f> corners_2d = {
        cv::Point2f(armor_bbox.x, armor_bbox.y), cv::Point2f(armor_bbox.x + armor_bbox.width, armor_bbox.y),
        cv::Point2f(armor_bbox.x, armor_bbox.y + armor_bbox.height),
        cv::Point2f(armor_bbox.x + armor_bbox.width, armor_bbox.y + armor_bbox.height)};
    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(corners_3d_, corners_2d, camera_matrix_, dist_coeffs_, rvec, tvec);
    if (success && !tvec.empty()) {
        double tx = tvec.at<double>(0);
        double ty = tvec.at<double>(1);
        double tz = tvec.at<double>(2);
        double yaw = std::atan2(tx, tz);
        double pitch = std::atan2(-ty, std::sqrt(tx * tx + tz * tz));
        double distance = std::sqrt(tx * tx + ty * ty + tz * tz);
        return {yaw, pitch, distance};
    }
    return {0.0, 0.0, 3.0};
}

/**
 * @brief TensorRT 引擎初始化
 *
 * 加载流程：
 *   1. 读取 .engine 文件到内存
 *   2. 创建 TRT runtime → 反序列化 CUDA engine → 创建执行上下文
 *   3. 查询输入/输出 tensor 形状（输入：3×H×W，输出：[5+class_num]×N）
 *   4. 分配 GPU 显存（输入 buffer + 输出 buffer）
 *   5. 创建 CUDA stream 用于异步推理
 *
 * 异常处理：任何步骤失败都会调用 cleanup() 释放已分配资源后重新抛出
 */
void YOLOv5Detector::initTensorRT() {
    std::ifstream file(engine_path_, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open engine: " + engine_path_);

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    if (size == 0) throw std::runtime_error("Engine file is empty");

    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("Failed to create TRT runtime");

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) throw std::runtime_error("Failed to deserialize CUDA engine");

    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create execution context");

    // 查询输入 tensor 形状：[batch, channel, height, width]
    auto input_dims = engine_->getTensorShape("images");
    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];

    // 查询输出 tensor 形状：[batch, bbox_features, num_detections]
    auto output_dims = engine_->getTensorShape("output0");
    output_dim1_ = output_dims.d[1];  // 5 + class_num（cx,cy,w,h,cls0,cls1）
    output_dim2_ = output_dims.d[2];  // 检测框数量

    // 分配 GPU 显存
    cudaError_t err;
    err = cudaMalloc(&gpu_buffers_[0], 3 * input_h_ * input_w_ * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("CUDA malloc input failed");
    size_t output_size = output_dim1_ * output_dim2_ * sizeof(float);
    err = cudaMalloc(&gpu_buffers_[1], output_size);
    if (err != cudaSuccess) {
        cudaFree(gpu_buffers_[0]);
        throw std::runtime_error("CUDA malloc output failed");
    }

    cpu_output_.resize(output_dim1_ * output_dim2_);

    if (cudaStreamCreate(&infer_stream_) != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cudaStreamCreate failed");
    }
}

/**
 * @brief 图像预处理
 * @param frame 原始 BGR 图像
 * @return 归一化后的 RGB 浮点图（3通道，值域 [0,1]）
 *
 * 步骤：resize → BGR2RGB → 转 float32 → 除以 255
 * 注意：YOLOv5 要求 RGB 输入，OpenCV 默认 BGR，必须转换
 */
cv::Mat YOLOv5Detector::preprocess(const cv::Mat& frame) {
    cv::Mat resized, rgb;
    cv::resize(frame, resized, cv::Size(input_w_, input_h_));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    return rgb;
}

/**
 * @brief TensorRT 异步推理
 * @param processed 预处理后的图像（HWC 格式）
 * @return 解码后的检测结果列表
 *
 * 数据布局转换：
 *   OpenCV 的 HWC 格式（交错）→ TRT 要求的 CHW 格式（平面）
 *   input_data = [R通道所有像素 | G通道所有像素 | B通道所有像素]
 *
 * 推理流程：
 *   H2D 拷贝输入 → 清零输出 buffer → 设置 tensor 地址 → 异步执行 → 同步等待 → D2H 拷贝输出
 */
std::vector<YOLOv5Detector::Detection> YOLOv5Detector::infer(const cv::Mat& processed) {
    // HWC → CHW 转换（复用预分配缓冲区）
    const float* src = processed.ptr<float>();
    int num_pixels = input_h_ * input_w_;
    for (int i = 0; i < num_pixels; i++) {
        input_data_[i] = src[i * 3];                       // R 通道
        input_data_[num_pixels + i] = src[i * 3 + 1];      // G 通道
        input_data_[2 * num_pixels + i] = src[i * 3 + 2];  // B 通道
    }

    // Host → Device
    if (cudaMemcpy(gpu_buffers_[0], input_data_.data(), 3 * input_h_ * input_w_ * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "[YOLO] cudaMemcpy H2D failed" << std::endl;
        return {};
    }
    if (cudaMemset(gpu_buffers_[1], 0, output_dim1_ * output_dim2_ * sizeof(float)) != cudaSuccess) {
        std::cerr << "[YOLO] cudaMemset failed" << std::endl;
        return {};
    }

    // 设置 tensor 地址并执行推理
    if (!context_->setTensorAddress("images", gpu_buffers_[0]) ||
        !context_->setTensorAddress("output0", gpu_buffers_[1])) {
        std::cerr << "[YOLO] setTensorAddress failed" << std::endl;
        return {};
    }
    bool ok = context_->enqueueV3(infer_stream_);
    if (!ok) {
        if (++infer_err_count_ <= 3) std::cerr << "[YOLO] enqueueV3 failed!" << std::endl;
        return {};
    }

    // 同步等待推理完成
    if (cudaStreamSynchronize(infer_stream_) != cudaSuccess) {
        std::cerr << "[YOLO] cudaStreamSynchronize failed" << std::endl;
        return {};
    }

    // Device → Host
    if (cudaMemcpy(cpu_output_.data(), gpu_buffers_[1], output_dim1_ * output_dim2_ * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "[YOLO] cudaMemcpy D2H failed" << std::endl;
        return {};
    }
    return decodeDetections();
}

/**
 * @brief 检测结果解码 + NMS 后处理
 *
 * YOLOv5 输出格式（每个检测框）：
 *   cpu_output_[col * output_dim2_ + row]，其中 row 是检测框索引
 *   [0] cx, [1] cy, [2] w, [3] h  —— 边界框（归一化坐标）
 *   [4] cls0_score, [5] cls1_score —— 两类置信度
 *
 * 解码步骤：
 *   1. 取两类中较大值作为 best_score，对应类别作为 best_cls
 *   2. 过滤：score < conf_threshold(0.1) 或类别不在目标范围内 → 丢弃
 *   3. 按 score 降序排序
 *   4. NMS（非极大值抑制）：IoU > 0.45 的重叠框被抑制
 */
std::vector<YOLOv5Detector::Detection> YOLOv5Detector::decodeDetections() {
    std::vector<Detection> detections;

    for (int i = 0; i < output_dim2_; i++) {
        float cx = cpu_output_[0 * output_dim2_ + i];
        float cy = cpu_output_[1 * output_dim2_ + i];
        float w = cpu_output_[2 * output_dim2_ + i];
        float h = cpu_output_[3 * output_dim2_ + i];
        float cls0 = cpu_output_[4 * output_dim2_ + i];
        float cls1 = cpu_output_[5 * output_dim2_ + i];
        float best_score = std::max(cls0, cls1);
        int best_cls = (cls0 > cls1) ? 0 : 1;

        // 过滤低置信度和非目标类别
        if (best_score < conf_threshold_) continue;
        if (best_cls < target_class_start_ || best_cls > target_class_end_) continue;

        Detection det;
        det.x = cx - w / 2;
        det.y = cy - h / 2;
        det.w = w;
        det.h = h;
        det.center_x = cx;
        det.center_y = cy;
        det.score = best_score;
        det.class_id = best_cls;
        detections.push_back(det);
    }

    // 按置信度降序排序
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<Detection> nms_result;
    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        nms_result.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            // 计算 IoU
            float ix = std::max(detections[i].x, detections[j].x);
            float iy = std::max(detections[i].y, detections[j].y);
            float ax = std::min(detections[i].x + detections[i].w, detections[j].x + detections[j].w);
            float ay = std::min(detections[i].y + detections[i].h, detections[j].y + detections[j].h);
            float ow = ax - ix, oh = ay - iy;
            if (ow <= 0 || oh <= 0) continue;
            float inter = ow * oh;
            float area_i = detections[i].w * detections[i].h;
            float area_j = detections[j].w * detections[j].h;
            float iou = inter / (area_i + area_j - inter);
            if (iou > nms_iou_threshold_) suppressed[j] = true;
        }
    }
    return nms_result;
}

/**
 * @brief 资源清理 —— 释放 GPU 显存和 CUDA stream
 * 在析构函数和 initTensorRT 失败时调用
 */
void YOLOv5Detector::cleanup() {
    if (gpu_buffers_[0]) {
        cudaFree(gpu_buffers_[0]);
        gpu_buffers_[0] = nullptr;
    }
    if (gpu_buffers_[1]) {
        cudaFree(gpu_buffers_[1]);
        gpu_buffers_[1] = nullptr;
    }
    cpu_output_.clear();
    if (infer_stream_) {
        cudaStreamDestroy(infer_stream_);
        infer_stream_ = nullptr;
    }
}
