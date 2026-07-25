/**
 * @file performance_monitor.h
 * @brief 性能监控器声明 —— 通用计时器 API，采集执行时间并计算统计指标
 *
 * PerformanceMonitor 提供 startTimer/endTimer pair 计时接口，
 * 自动计算每个 metric 的平均值、最小/最大值、标准差。
 *
 * 线程安全：所有公开方法加 mutex_ 保护，可在多个定时器线程中安全调用。
 * 内存管理：每个 metric 最多保留 max_samples_ 条记录（默认100），超出后 FIFO 淘汰。
 *
 * 使用方式：
 *   perf_monitor->startTimer("inference");
 *   // ... 执行推理 ...
 *   perf_monitor->endTimer("inference");
 *   // 定期调用 reportAll() 输出报告
 *
 * 使用者：YOLOv5Detector, CombatManager, MovementController, StateManager
 */

#ifndef PERFORMANCE_MONITOR_H
#define PERFORMANCE_MONITOR_H

#include <deque>
#include <string>
#include <chrono>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <rclcpp/rclcpp.hpp>

/**
 * @brief 性能指标快照 —— 存储单个 metric 的统计结果
 */
struct PerformanceMetrics {
    std::string name;       ///< 指标名称
    double avg_ms;          ///< 平均执行时间（毫秒）
    double min_ms;          ///< 最小执行时间
    double max_ms;          ///< 最大执行时间
    double std_dev_ms;      ///< 标准差（反映波动程度）
    size_t sample_count;    ///< 总采样次数（不受滑动窗口影响）
};

class PerformanceMonitor {
private:
    /**
     * @brief 单个指标的内部数据 —— 存储原始采样值和计时状态
     */
    struct MetricData {
        std::string name;                                        ///< 指标名称
        std::deque<double> values;                               ///< 采样值队列（滑动窗口）
        std::chrono::steady_clock::time_point start_time;        ///< 本次计时起始点
        size_t total_count;                                      ///< 总调用次数
        MetricData(const std::string& n);
    };

    mutable std::mutex mutex_;                                           ///< 互斥锁
    std::unordered_map<std::string, std::unique_ptr<MetricData>> metrics_;  ///< 所有指标
    size_t max_samples_;                                                ///< 每个 metric 最大采样数
    rclcpp::Logger logger_;                                             ///< ROS2 日志器

    /** @brief 计算标准差（总体标准差，除以 N） */
    double calculateStdDev(const std::deque<double>& values, double mean) const;

    /** @brief 计算平均值 */
    double calculateMean(const std::deque<double>& values) const;

    /** @brief 获取统计指标（内部方法，需持有锁） */
    PerformanceMetrics getMetricsUnsafe(const std::string& metric_name) const;

public:
    /**
     * @brief 构造函数
     * @param max_samples 每个 metric 保留的最大采样数（默认100）
     * @param logger      ROS2 日志器
     */
    PerformanceMonitor(size_t max_samples = 100,
                       const rclcpp::Logger& logger = rclcpp::get_logger("PerformanceMonitor"));

    /** @brief 启动计时（记录起始时间点，首次调用自动创建 metric） */
    void startTimer(const std::string& metric_name);

    /** @brief 停止计时并记录耗时（自动裁剪超出窗口的旧数据） */
    void endTimer(const std::string& metric_name);

    /** @brief 输出所有 metric 的性能报告到 ROS2 日志 */
    void reportAll();
};

#endif
