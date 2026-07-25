/**
 * @file target_ekf.h
 * @brief 目标扩展卡尔曼滤波器声明 —— 跟踪目标的 yaw/pitch/distance 三状态
 *
 * TargetEKF 封装 GeneralKalmanFilter，专门用于跟踪单个目标的云台指向和距离。
 * 状态向量 x = [yaw, pitch, distance]^T（3维）
 *
 * 运动模型：随机游走模型（F = I）。当 q_adaptive_alpha > 0 时，
 * 根据归一化 innovation（sqrt(NIS/NX)）自适应缩放 Q。
 *
 * 距离测量噪声自适应：
 *   R_distance = log(d+1)/denom + base（对数模型，远距离噪声更大）
 *
 * 使用场景：
 *   CombatManager 中的 target_ekf_ 实例，每帧调用 update() 融合测量，
 *   射击前调用 predict() 外推目标位置（补偿通信延迟）。
 */

#ifndef COMPETITION_TARGET_EKF_H
#define COMPETITION_TARGET_EKF_H

#include "general_kalman_filter.h"

struct TargetEKFConfig {
    double q_yaw = 0.01, q_pitch = 0.01, q_distance = 0.5;
    double r_yaw = 4e-3, r_pitch = 4e-3;
    double r_distance_denom = 200.0, r_distance_base = 0.09;
    double p0_yaw = 1e-2, p0_pitch = 1e-2, p0_distance = 1e-1;
    double p_reset = 1e3;
    double q_adaptive_alpha = 0.0;
    double q_adaptive_max_scale = 10.0;
    double chi2_upper = 7.815;
    int nis_window = 100;
};

class TargetEKF {
public:
    explicit TargetEKF(const TargetEKFConfig& cfg = {});

    std::tuple<double, double, double> update(double dt, double meas_yaw, double meas_pitch, double meas_distance);

    std::tuple<double, double, double> predict(double dt);

    void reset();

    double nisFailureRate() const { return ekf_.nisFailureRate(); }

private:
    static constexpr int NX = 3;

    TargetEKFConfig cfg_;
    GeneralKalmanFilter ekf_;
    bool initialized_ = false;
    double last_normalized_innovation_ = 0.0;

    double adaptiveQScale() const;
};

#endif
