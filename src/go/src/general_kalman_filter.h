/**
 * @file general_kalman_filter.h
 * @brief 通用扩展卡尔曼滤波器（EKF）声明 —— 与具体运动模型解耦的滤波框架
 *
 * 本类实现了标准卡尔曼滤波的 predict/update 步骤，通过函数对象适配非线性模型：
 *   - h: 观测函数（非线性映射 x → z）
 *   - z_subtract: 观测残差计算（处理角度环绕等非线性差值）
 *
 * NIS（Normalized Innovation Squared）检验：
 *   每次 update() 后计算 NIS，维护滑动窗口内的失败率（nisFailureRate），
 *   用于检测滤波器是否发散（测量与预测严重不匹配）。
 *
 * 当前唯一使用者：TargetEKF（状态维度 NX=3：yaw, pitch, distance）
 *
 * 约瑟夫形式协方差更新：
 *   P = (I-KH)P(I-KH)^T + KRK^T（保证对称正定，比简化形式更稳定）
 */

#ifndef COMPETITION_GENERAL_KALMAN_FILTER_H
#define COMPETITION_GENERAL_KALMAN_FILTER_H

#include <deque>
#include <functional>

#include <opencv2/core.hpp>

class GeneralKalmanFilter {
public:
    cv::Mat x;  ///< 状态向量（dim × 1）
    cv::Mat P;  ///< 状态协方差矩阵（dim × dim）

    /**
     * @brief 构造函数
     * @param dim        状态维度（TargetEKF 中为 3）
     * @param chi2_upper NIS 卡方检验上界（自由度=dim）
     * @param nis_window NIS 失败率统计窗口大小
     */
    GeneralKalmanFilter(int dim, double chi2_upper, int nis_window);

    /**
     * @brief 预测步骤：x̂ = F·x, P = F·P·F^T + Q
     * @param F 状态转移矩阵
     * @param Q 过程噪声协方差
     * @return 预测后的状态向量
     */
    cv::Mat predict(const cv::Mat& F, const cv::Mat& Q);

    /**
     * @brief 更新步骤（带 NIS 检验）
     * @param z          观测向量
     * @param H          观测矩阵
     * @param R          观测噪声协方差
     * @param h          观测函数（非线性映射）
     * @param z_subtract 观测残差计算（处理角度环绕）
     * @return NIS 值
     */
    double update(const cv::Mat& z, const cv::Mat& H, const cv::Mat& R, const std::function<cv::Mat(const cv::Mat&)>& h,
                  const std::function<cv::Mat(const cv::Mat&, const cv::Mat&)>& z_subtract);

    /** @brief 获取最近一次 NIS 值 */
    double nis() const { return last_nis_; }

    /** @brief 获取滑动窗口内的 NIS 失败率（0.0~1.0） */
    double nisFailureRate() const;

    /** @brief 重置状态和协方差（目标丢失/切换时调用） */
    void reset(const cv::Mat& x0, const cv::Mat& P0);

private:
    int dim_;                              ///< 状态维度
    cv::Mat I_;                            ///< 单位矩阵（dim × dim）
    double chi2_upper_;                    ///< NIS 卡方检验上界
    int nis_window_;                       ///< NIS 统计窗口大小
    double last_nis_ = 0.0;                ///< 最近一次 NIS 值
    std::deque<int> recent_nis_failures_;  ///< 滑动窗口内的 NIS 失败记录
};

#endif
