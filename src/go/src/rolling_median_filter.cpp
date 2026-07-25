/**
 * @file rolling_median_filter.cpp
 * @brief 滑动窗口中值滤波器 —— 平滑距离测量值，消除异常跳变
 *
 * 为什么用中值滤波而不是均值滤波？
 *   - 中值滤波对异常值（outlier）完全免疫：单个错误测量不会影响输出
 *   - 均值滤波会被一个极端值严重拉偏（如 PnP 返回距离=0 的错误帧）
 *   - 装甲板检测的距离估计偶尔会出现跳变（遮挡/误检），中值滤波可以平滑掉
 *
 * 工作原理：
 *   维护一个固定大小的 deque 作为滑动窗口，每次新数据 push_back，
 *   超出窗口大小时 pop_front。取窗口内所有值的中位数作为输出。
 *
 * 时间复杂度：O(N log N)（每帧排序一次，N = window_size，通常 5-10）
 * 空间复杂度：O(N)
 *
 * 使用场景：
 *   YOLOv5Detector 在 calculateBallisticCompensation() 中，
 *   将 PnP 测距结果通过此滤波器平滑后再用于弹道补偿计算。
 *   距离值的稳定性直接影响云台 pitch 角度的精度。
 */

#include "rolling_median_filter.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

/**
 * @brief 构造函数
 * @param window_size 滑动窗口大小（帧数），建议值 5-10
 *                    太小 → 平滑效果弱，异常值仍会影响
 *                    太大 → 响应延迟，目标距离快速变化时跟踪滞后
 */
RollingMedianFilter::RollingMedianFilter(int window_size) : window_size_(window_size) {
    if (window_size_ <= 0) {
        throw std::invalid_argument("RollingMedianFilter window_size must be positive");
    }
}

/**
 * @brief 输入新的测量值
 * @param value 本次测量值（如 PnP 解算出的距离）
 *
 * 超出窗口大小时自动删除最早的数据（FIFO），
 * 保证窗口内始终最多 window_size_ 条记录
 */
void RollingMedianFilter::update(double value) {
    buffer_.push_back(value);
    while (static_cast<int>(buffer_.size()) > window_size_) {
        buffer_.pop_front();
    }
}

/**
 * @brief 获取滤波后的值（窗口内中位数）
 * @return 中位数，空窗口返回 0.0
 *
 * 实现方式：拷贝 buffer 到 vector → 排序 → 取中间元素
 *   - 窗口大小为奇数时：取正中间的值
 *   - 窗口大小为偶数时：取中间偏右的值（size/2）
 *   - 窗口为空时返回 0.0（安全默认值）
 */
double RollingMedianFilter::getFiltered() const {
    if (buffer_.empty()) return 0.0;
    std::vector<double> sorted(buffer_.begin(), buffer_.end());
    auto mid = sorted.begin() + sorted.size() / 2;
    std::nth_element(sorted.begin(), mid, sorted.end());
    return *mid;
}

/** @brief 清空缓冲区（如目标切换时调用，避免用旧数据滤波新目标） */
void RollingMedianFilter::reset() { buffer_.clear(); }
