/**
 * @file rolling_median_filter.h
 * @brief 滑动窗口中值滤波器声明 —— 平滑距离测量值，消除 PnP 异常跳变
 *
 * 为什么用中值滤波？
 *   - PnP 单目测距偶尔会返回极端错误值（如距离=0 或距离=100）
 *   - 中值滤波对异常值完全免疫（单个极端值不影响中位数）
 *   - 均值滤波会被一个极端值严重拉偏
 *
 * 默认窗口大小 150 帧（约5秒 @30fps），足以平滑短时波动，
 * 同时对目标距离的持续变化保持响应。
 *
 * 使用场景：
 *   YOLOv5Detector 在每次 PnP 测距后调用 update() + getFiltered()，
 *   平滑后的距离用于弹道补偿计算。
 */

#ifndef COMPETITION_ROLLING_MEDIAN_FILTER_H
#define COMPETITION_ROLLING_MEDIAN_FILTER_H

#include <deque>

class RollingMedianFilter {
public:
    /**
     * @brief 构造函数
     * @param window_size 滑动窗口大小（帧数），默认150
     */
    explicit RollingMedianFilter(int window_size = 150);

    /** @brief 输入新的测量值，超出窗口时自动删除最早数据 */
    void update(double value);

    /** @brief 获取窗口内中位数（滤波后的值），空窗口返回 0.0 */
    double getFiltered() const;

    /** @brief 清空缓冲区（目标切换时调用） */
    void reset();

private:
    int window_size_;              ///< 滑动窗口大小
    std::deque<double> buffer_;    ///< 数据缓冲区（FIFO）
};

#endif
