/**
 * @file target_ekf.cpp
 * @brief 目标扩展卡尔曼滤波器 —— 跟踪目标的 yaw/pitch/distance 三状态
 *
 * TargetEKF 是 GeneralKalmanFilter 的具体实例化，专门用于跟踪单个目标的云台指向角度和距离。
 * 状态向量 x = [yaw, pitch, distance]^T（3维）
 *
 * 为什么需要 EKF 而不是直接用测量值？
 *   - 相机检测存在噪声（尤其是距离估计）
 *   - 目标可能短暂丢失（遮挡/出视野），需要预测其位置
 *   - 弹道补偿需要平滑的角度和距离输入
 *
 * 运动模型：随机游走模型（F = I），假设目标在短时间内状态变化较小。
 * 这对于近距离固定目标（如前哨站/基地）是合理的近似。
 *
 * 自适应过程噪声（q_adaptive_alpha > 0 时启用）：
 *   当 innovation（测量与预测之差）偏大时，说明匀速假设不成立（目标在加速/转弯），
 *   此时放大 Q 让滤波器更信任测量值，减小跟踪滞后。
 *   使用 sqrt(NIS/NX) 统一角度和距离的量纲，避免距离残差主导缩放。
 *   公式：Q_scaled = Q * min(max_scale, 1 + α * sqrt(NIS/NX))
 *   α = 0 时退化为固定 Q 模型。
 *
 * 距离测量噪声自适应（始终启用）：
 *   R_distance = log(d+1) / R_denom + R_base（对数模型，远距离噪声更大）
 *   原因：PnP 单目测距的误差随距离增长，远距离时更依赖预测。
 *
 * 角度残差处理：
 *   yaw/pitch 是循环角度，atan2(sin(Δ), cos(Δ)) 归一化到 [-π, π]。
 */

#include "target_ekf.h"

#include <algorithm>
#include <cmath>

TargetEKF::TargetEKF(const TargetEKFConfig& cfg) : cfg_(cfg), ekf_(NX, cfg_.chi2_upper, cfg_.nis_window) { reset(); }

double TargetEKF::adaptiveQScale() const {
    if (cfg_.q_adaptive_alpha <= 0.0) return 1.0;
    const double max_scale = std::max(1.0, cfg_.q_adaptive_max_scale);
    return std::clamp(1.0 + cfg_.q_adaptive_alpha * last_normalized_innovation_, 1.0, max_scale);
}

/**
 * @brief EKF 更新步骤（接收新测量值，融合预测与观测）
 * @param dt            距上次更新的时间间隔（秒）
 * @param meas_yaw      测量的水平角度
 * @param meas_pitch    测量的俯仰角度
 * @param meas_distance 测量的距离（地图坐标单位）
 * @return 滤波后的 {yaw, pitch, distance} 元组
 *
 * 首次调用时直接用测量值初始化状态（不经过滤波）。
 *
 * 过程噪声 Q 的设计：
 *   q_yaw/pitch = 0.01：角度变化缓慢（目标是固定建筑）
 *   q_distance  = 0.5：距离变化相对较大
 *   Q 随 dt 线性增长：时间越长，预测不确定性越大
 *   q_adaptive_alpha > 0 时 Q 按归一化 innovation 自动缩放并限制最大倍率
 *
 * 观测噪声 R 的设计：
 *   角度噪声固定为 4e-3（相机角度测量精度稳定）
 *   距离噪声自适应：R_dist = log(d+1)/200 + 0.09
 *     - d=0 时：R ≈ 0.09（近距离，精度高）
 *     - d=10 时：R ≈ 0.10（中距离）
 *     - d=30 时：R ≈ 0.12（远距离，精度下降）
 *     使用 log 而非线性是因为 PnP 误差增长是亚线性的
 */
std::tuple<double, double, double> TargetEKF::update(double dt, double meas_yaw, double meas_pitch,
                                                     double meas_distance) {
    // 首次观测：直接初始化状态，跳过滤波
    if (!initialized_) {
        cv::Mat x0(NX, 1, CV_64F);
        x0.at<double>(0) = meas_yaw;
        x0.at<double>(1) = meas_pitch;
        x0.at<double>(2) = meas_distance;

        cv::Mat P0 = cv::Mat::eye(NX, NX, CV_64F);
        P0.at<double>(0, 0) = cfg_.p0_yaw;
        P0.at<double>(1, 1) = cfg_.p0_pitch;
        P0.at<double>(2, 2) = cfg_.p0_distance;

        ekf_.reset(x0, P0);
        initialized_ = true;

        return {meas_yaw, meas_pitch, meas_distance};
    }

    // ── 预测阶段 ──
    cv::Mat F = cv::Mat::eye(NX, NX, CV_64F);

    cv::Mat Q = cv::Mat::eye(NX, NX, CV_64F);
    Q.at<double>(0, 0) = cfg_.q_yaw * dt;
    Q.at<double>(1, 1) = cfg_.q_pitch * dt;
    Q.at<double>(2, 2) = cfg_.q_distance * dt;

    // 使用上次更新的归一化 innovation 调整本次过程噪声。
    Q *= adaptiveQScale();

    ekf_.predict(F, Q);

    // ── 更新阶段 ──
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    R.at<double>(0, 0) = cfg_.r_yaw;
    R.at<double>(1, 1) = cfg_.r_pitch;
    R.at<double>(2, 2) = std::log(meas_distance + 1.0) / cfg_.r_distance_denom + cfg_.r_distance_base;

    // 观测向量
    cv::Mat z(3, 1, CV_64F);
    z.at<double>(0) = meas_yaw;
    z.at<double>(1) = meas_pitch;
    z.at<double>(2) = meas_distance;

    // 观测矩阵 H = I（直接观测状态本身，无非线性映射）
    cv::Mat H = cv::Mat::eye(NX, NX, CV_64F);
    auto h = [](const cv::Mat& x) -> cv::Mat { return x; };

    // 残差计算：角度使用 atan2 归一化到 [-π, π]，防止环绕跳变
    auto z_subtract = [](const cv::Mat& a, const cv::Mat& b) -> cv::Mat {
        cv::Mat c = a - b;
        c.at<double>(0) = std::atan2(std::sin(c.at<double>(0)), std::cos(c.at<double>(0)));
        c.at<double>(1) = std::atan2(std::sin(c.at<double>(1)), std::cos(c.at<double>(1)));
        return c;
    };

    const double nis = ekf_.update(z, H, R, h, z_subtract);
    last_normalized_innovation_ = std::sqrt(std::max(0.0, nis) / NX);

    return {ekf_.x.at<double>(0), ekf_.x.at<double>(1), ekf_.x.at<double>(2)};
}

/**
 * @brief 无观测时的纯预测（用于射击前的目标位置外推）
 * @param dt 距上次 update/predict 的时间间隔（秒）
 * @return 预测的 {yaw, pitch, distance} 元组
 *
 * 使用场景：检测到目标后到实际开枪之间有时间延迟
 * （通信延迟、弹丸飞行时间等），需要 predict() 外推目标当前位置。
 * 未初始化时返回 {0, 0, 3.0}（安全默认值，避免指向错误方向）。
 */
std::tuple<double, double, double> TargetEKF::predict(double dt) {
    if (!initialized_) return {0.0, 0.0, 3.0};

    cv::Mat F = cv::Mat::eye(NX, NX, CV_64F);

    cv::Mat Q = cv::Mat::eye(NX, NX, CV_64F);
    Q.at<double>(0, 0) = cfg_.q_yaw * dt;
    Q.at<double>(1, 1) = cfg_.q_pitch * dt;
    Q.at<double>(2, 2) = cfg_.q_distance * dt;

    Q *= adaptiveQScale();

    ekf_.predict(F, Q);

    return {ekf_.x.at<double>(0), ekf_.x.at<double>(1), ekf_.x.at<double>(2)};
}

/**
 * @brief 重置滤波器（目标丢失/切换时调用）
 *
 * 状态重置为全零，协方差重置为 p_reset * I（表示完全不确定）。
 * initialized_ = false，下次 update() 会用测量值直接初始化。
 *
 * 重置时机：
 *   - 检测框中心距上一帧超过 100 像素（目标跳变）
 *   - 连续多帧未检测到目标后重新出现
 *   - NIS 失败率持续过高（滤波器发散）
 */
void TargetEKF::reset() {
    cv::Mat x0 = cv::Mat::zeros(NX, 1, CV_64F);
    cv::Mat P0 = cv::Mat::eye(NX, NX, CV_64F) * cfg_.p_reset;
    ekf_.reset(x0, P0);
    initialized_ = false;
    last_normalized_innovation_ = 0.0;
}
