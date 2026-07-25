/**
 * @file general_kalman_filter.cpp
 * @brief 通用扩展卡尔曼滤波器（EKF）—— 预测、更新、NIS 有效性检验
 *
 * 本实现是一个与具体运动模型解耦的通用 EKF 框架，通过模板化的状态转移矩阵 F、
 * 观测函数 h、残差计算 z_subtract 来适配不同的跟踪场景。
 *
 * 当前唯一使用者：TargetEKF（目标云台角度+距离跟踪）
 * 状态向量 x = [yaw, pitch, distance]^T（3维）
 *
 * 标准卡尔曼滤波流程：
 *   预测阶段：
 *     x̂ = F · x        （状态预测）
 *     P  = F · P · F^T + Q （协方差预测）
 *
 *   更新阶段：
 *     S = H · P · H^T + R         （新息协方差）
 *     K = P · H^T · S⁻¹           （卡尔曼增益）
 *     P = (I - KH) · P · (I - KH)^T + K · R · K^T （约瑟夫形式，数值更稳定）
 *     x = x + K · (z - h(x))      （状态更新）
 *
 * NIS（Normalized Innovation Squared）检验：
 *   NIS = r^T · S⁻¹ · r，其中 r = z - h(x) 是新息（残差）
 *   NIS 服从卡方分布，如果 NIS 持续过大，说明：
 *     - 观测数据与预测模型不匹配（目标机动/丢失）
 *     - 滤波器可能发散
 *   本模块维护一个滑动窗口内的 NIS 失败率（nisFailureRate），
 *   供 TargetEKF 决定是否需要重置滤波器。
 *
 * 数值稳定性措施：
 *   - 使用约瑟夫形式更新 P（而非简化形式 P = (I-KH)P）
 *   - 角度残差使用 atan2(sin, cos) 处理 ±π 环绕
 *   - 提供 reset() 方法重新初始化状态和协方差
 */

#include "general_kalman_filter.h"

#include <cmath>
#include <numeric>

/**
 * @brief 构造函数 —— 初始化状态维度、NIS 阈值和协方差矩阵
 * @param dim       状态向量维度（TargetEKF 中为 3：yaw, pitch, distance）
 * @param chi2_upper NIS 卡方检验上界（自由度=dim），超过则视为测量异常
 * @param nis_window NIS 失败率统计的滑动窗口大小
 *
 * 初始状态：x = 零向量，P = 单位矩阵（表示对初始状态完全不确定）
 */
GeneralKalmanFilter::GeneralKalmanFilter(int dim, double chi2_upper, int nis_window)
    : dim_(dim),
      chi2_upper_(chi2_upper),
      nis_window_(nis_window),
      I_(cv::Mat::eye(dim, dim, CV_64F)),
      x(cv::Mat::zeros(dim, 1, CV_64F)),
      P(cv::Mat::eye(dim, dim, CV_64F)) {}

/**
 * @brief 卡尔曼滤波预测步骤
 * @param F 状态转移矩阵（随机游走模型下为单位阵，即期望状态不变）
 * @param Q 过程噪声协方差矩阵（对角阵，各状态独立的噪声强度）
 * @return 预测后的状态向量 x̂
 *
 * 注意：本实现是纯预测（不涉及非线性变换），所以 F 是线性矩阵。
 * 在随机游走模型下 F = I（期望状态不变），Q 决定了"允许状态变化多快"：
 *   q_yaw 越大 → 允许 yaw 变化越快（跟踪机动目标）
 *   q_distance 越大 → 允许距离变化越快（目标快速靠近/远离）
 */
cv::Mat GeneralKalmanFilter::predict(const cv::Mat& F, const cv::Mat& Q) {
    P = F * P * F.t() + Q;
    x = F * x;
    return x;
}

/**
 * @brief 卡尔曼滤波更新步骤（带 NIS 检验）
 * @param z          观测向量（TargetEKF 中为 [meas_yaw, meas_pitch, meas_distance]^T）
 * @param H          观测矩阵（线性观测时为单位阵）
 * @param R          观测噪声协方差矩阵（对角阵，各观测量的噪声方差）
 * @param h          观测函数（非线性观测时使用，线性时为恒等函数）
 * @param z_subtract 观测残差计算函数（处理角度环绕等非线性差值）
 * @return NIS 值（归一化新息平方）
 *
 * 更新流程：
 *   1. 计算新息协方差 S 和卡尔曼增益 K
 *   2. 使用约瑟夫形式更新 P（数值更稳定，保证 P 正定）
 *   3. 使用 z_subtract 计算非线性残差，更新状态 x
 *   4. 在状态更新前用预测残差和预测 S 计算标准 NIS
 *   5. 将 NIS 与 chi2_upper 比较，记录到滑动窗口
 *
 * 约瑟夫形式 vs 简化形式：
 *   简化：P = (I - KH)P（可能因浮点误差导致 P 不对称/非正定）
 *   约瑟夫：P = (I-KH)P(I-KH)^T + KRK^T（保证对称正定，代价是多一次矩阵乘法）
 */
double GeneralKalmanFilter::update(const cv::Mat& z, const cv::Mat& H, const cv::Mat& R,
                                   const std::function<cv::Mat(const cv::Mat&)>& h,
                                   const std::function<cv::Mat(const cv::Mat&, const cv::Mat&)>& z_subtract) {
    // 预测新息和新息协方差必须在状态更新前计算，才能得到标准 NIS。
    cv::Mat residual = z_subtract(z, h(x));
    cv::Mat S = H * P * H.t() + R;
    cv::Mat S_inv = S.inv();
    last_nis_ = cv::Mat(residual.t() * S_inv * residual).at<double>(0);

    // 卡尔曼增益 K = PH^T S⁻¹
    cv::Mat K = P * H.t() * S_inv;

    // 约瑟夫形式更新协方差：P = (I-KH)P(I-KH)^T + KRK^T
    cv::Mat IKH = I_ - K * H;
    P = IKH * P * IKH.t() + K * R * K.t();

    // 状态更新：x = x + K * (z - h(x))
    // z_subtract 处理角度环绕（如 yaw 从 359° 到 1° 的差值应为 2° 而非 -358°）
    x = x + K * residual;

    // 记录 NIS 是否超过卡方阈值（滑动窗口统计）
    bool nis_fail = last_nis_ > chi2_upper_;
    recent_nis_failures_.push_back(nis_fail ? 1 : 0);
    if (static_cast<int>(recent_nis_failures_.size()) > nis_window_) recent_nis_failures_.pop_front();

    return last_nis_;
}

/**
 * @brief 计算滑动窗口内的 NIS 失败率
 * @return 失败率（0.0~1.0），即最近 nis_window 次更新中 NIS 超标的比例
 *
 * 使用场景：TargetEKF 在每次 update 后检查此值，
 * 如果失败率过高（如 > 0.5），说明滤波器可能发散，需要 reset
 */
double GeneralKalmanFilter::nisFailureRate() const {
    if (recent_nis_failures_.empty()) return 0.0;
    int failures = std::accumulate(recent_nis_failures_.begin(), recent_nis_failures_.end(), 0);
    return static_cast<double>(failures) / recent_nis_failures_.size();
}

/**
 * @brief 重置滤波器状态和协方差
 * @param x0 初始状态向量（TargetEKF 中为全零）
 * @param P0 初始协方差矩阵（TargetEKF 中为 1000*I，表示高度不确定）
 *
 * 重置时机：
 *   - 目标丢失后重新检测到
 *   - NIS 失败率持续过高
 *   - 检测到目标身份切换（如从一个装甲板跳到另一个）
 */
void GeneralKalmanFilter::reset(const cv::Mat& x0, const cv::Mat& P0) {
    x0.copyTo(x);
    P0.copyTo(P);
    recent_nis_failures_.clear();
    last_nis_ = 0.0;
}
