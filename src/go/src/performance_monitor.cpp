/**
 * @file performance_monitor.cpp
 * @brief 性能监控器 —— 采集各模块执行时间，计算统计指标，输出性能报告
 *
 * PerformanceMonitor 提供了一套通用的计时器 API，任意代码段都可以通过
 * startTimer/endTimer  pair 记录执行耗时。
 *
 * 典型使用场景（在各模块的定时器回调中）：
 *   performance_monitor->startTimer("detection_loop");
 *   // ... 执行检测 ...
 *   performance_monitor->endTimer("detection_loop");
 *
 * 采集的数据存储在 MetricData 中（per-metric 的 deque），
 * 支持以下统计查询：
 *   - 平均值（mean）：反映典型执行时间
 *   - 最小/最大值：反映最坏/最好情况
 *   - 标准差（std_dev）：反映执行时间的波动程度
 *   - 总调用次数：反映执行频率
 *
 * 内存管理：
 *   每个 metric 最多保留 max_samples_ 条记录（默认100），
 *   超出后删除最早的数据（FIFO 滑动窗口）。
 *   这保证了内存使用恒定，不会因长时间运行而膨胀。
 *
 * 线程安全：
 *   所有公开方法加 mutex_ 保护，可在多个定时器线程中安全调用。
 *   getMetricsUnsafe 是内部方法，调用方需自行保证锁。
 *
 * 使用者：
 *   - YOLOv5Detector：监控 TensorRT 推理耗时
 *   - CombatManager：监控检测+弹道计算总耗时
 *   - MovementController：监控运动控制循环耗时
 *   - StateManager：监控健康检查耗时
 */

#include "performance_monitor.h"
#include <vector>
#include <algorithm>
#include <cmath>

/** @brief MetricData 构造：初始化名称和计数器 */
PerformanceMonitor::MetricData::MetricData(const std::string& n) : name(n), total_count(0) {}

/**
 * @brief 构造函数
 * @param max_samples 每个 metric 保留的最大采样数（超出后删除最旧数据）
 * @param logger      ROS2 日志器，用于 reportAll() 输出
 */
PerformanceMonitor::PerformanceMonitor(size_t max_samples,
                                        const rclcpp::Logger& logger)
    : max_samples_(max_samples), logger_(logger) {}

/**
 * @brief 计算标准差
 * @param values 数据队列
 * @param mean   已计算的平均值（避免重复计算）
 * @return 标准差，数据不足2条返回 0
 *
 * 使用总体标准差公式（除以 N 而非 N-1），
 * 因为这里描述的是样本自身的波动，而非推断总体
 */
double PerformanceMonitor::calculateStdDev(const std::deque<double>& values, double mean) const {
    if (values.size() < 2) return 0.0;
    double sum_sq = 0.0;
    for (double v : values) {
        double diff = v - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / values.size());
}

/**
 * @brief 计算平均值
 * @param values 数据队列
 * @return 平均值，空队列返回 0
 */
double PerformanceMonitor::calculateMean(const std::deque<double>& values) const {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / values.size();
}

/**
 * @brief 启动计时器（记录起始时间点）
 * @param metric_name 指标名称（如 "detection_loop"、"ekf_update"）
 *
 * 首次调用某 metric_name 时自动创建 MetricData。
 * 使用 steady_clock（不受系统时间调整影响）保证计时准确。
 */
void PerformanceMonitor::startTimer(const std::string& metric_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (metrics_.find(metric_name) == metrics_.end()) {
        metrics_[metric_name] = std::make_unique<MetricData>(metric_name);
    }
    metrics_[metric_name]->start_time = std::chrono::steady_clock::now();
}

/**
 * @brief 停止计时器并记录耗时
 * @param metric_name 指标名称（必须先调用过 startTimer）
 *
 * 计算流程：
 *   1. 读取当前时间，与 start_time 做差得到耗时（微秒精度）
 *   2. 转换为毫秒存入 values 队列
 *   3. 递增 total_count（不受滑动窗口影响，记录总调用次数）
 *   4. 如果 values 超过 max_samples_，删除最旧的数据
 *
 * 注意：如果 startTimer 未被调用，会打 WARN 日志但不崩溃
 */
void PerformanceMonitor::endTimer(const std::string& metric_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(metric_name);
    if (it == metrics_.end()) {
        RCLCPP_WARN(logger_, "Metric '%s' not found. Call startTimer() first.", metric_name.c_str());
        return;
    }
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - it->second->start_time).count();
    double duration_ms = duration / 1000.0;
    it->second->values.push_back(duration_ms);
    it->second->total_count++;
    // 滑动窗口：超出 max_samples_ 时删除最旧数据
    if (it->second->values.size() > max_samples_) {
        it->second->values.erase(it->second->values.begin(),
                                 it->second->values.begin() + (it->second->values.size() - max_samples_));
    }
}

/**
 * @brief 获取指定 metric 的统计指标（内部方法，需持有锁）
 * @param metric_name 指标名称
 * @return PerformanceMetrics 结构体，包含 avg/min/max/std_dev/sample_count
 *
 * 空数据时返回全零结构体（不返回 NaN/异常值，简化调用方处理）
 */
PerformanceMetrics PerformanceMonitor::getMetricsUnsafe(const std::string& metric_name) const {
    auto it = metrics_.find(metric_name);
    if (it == metrics_.end()) {
        return PerformanceMetrics{metric_name, 0.0, 0.0, 0.0, 0.0, 0};
    }
    const auto& data = it->second;
    if (data->values.empty()) {
        return PerformanceMetrics{metric_name, 0.0, 0.0, 0.0, 0.0, data->total_count};
    }
    double mean = calculateMean(data->values);
    double min_val = *std::min_element(data->values.begin(), data->values.end());
    double max_val = *std::max_element(data->values.begin(), data->values.end());
    double std_dev = calculateStdDev(data->values, mean);
    return PerformanceMetrics{metric_name, mean, min_val, max_val, std_dev, data->total_count};
}

/**
 * @brief 输出所有 metric 的性能报告到 ROS2 日志
 *
 * 报告格式：
 *   ========== 性能监控报告 ==========
 *   metric_name: 无数据                          （无采样的 metric）
 *   metric_name: avg=X.XXms, min=X.XXms, max=X.XXms, std=X.XXms (采样: N)
 *   =================================
 *
 * 线程安全：在锁内复制数据快照后释放锁，再打日志（避免持锁打日志导致延迟）
 */
void PerformanceMonitor::reportAll() {
    std::vector<PerformanceMetrics> snapshots;
    std::vector<std::string> empty_metrics;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [name, data] : metrics_) {
            if (data->values.empty()) {
                empty_metrics.push_back(name);
            } else {
                double mean = calculateMean(data->values);
                double min_val = *std::min_element(data->values.begin(), data->values.end());
                double max_val = *std::max_element(data->values.begin(), data->values.end());
                double std_dev = calculateStdDev(data->values, mean);
                snapshots.push_back({name, mean, min_val, max_val, std_dev, data->total_count});
            }
        }
    }
    RCLCPP_INFO(logger_, "========== 性能监控报告 ==========");
    for (const auto& name : empty_metrics) {
        RCLCPP_INFO(logger_, "%s: 无数据", name.c_str());
    }
    for (const auto& m : snapshots) {
        RCLCPP_INFO(logger_,
                   "%s: avg=%.2fms, min=%.2fms, max=%.2fms, std=%.2fms (采样: %zu)",
                   m.name.c_str(),
                   m.avg_ms,
                   m.min_ms,
                   m.max_ms,
                   m.std_dev_ms,
                   m.sample_count);
    }
    RCLCPP_INFO(logger_, "=================================");
}
