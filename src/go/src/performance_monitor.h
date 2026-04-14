#ifndef PERFORMANCE_MONITOR_H
#define PERFORMANCE_MONITOR_H

#include <deque>
#include <string>
#include <chrono>
#include <cmath>
#include <rclcpp/rclcpp.hpp>

// 性能指标结构
struct PerformanceMetrics {
    std::string name;
    double avg_ms;
    double min_ms;
    double max_ms;
    double std_dev_ms;
    size_t sample_count;
};

// 性能监控器类
class PerformanceMonitor {
private:
    struct MetricData {
        std::string name;
        std::deque<double> values;
        std::chrono::steady_clock::time_point start_time;
        size_t total_count;
        
        MetricData(const std::string& n) : name(n), total_count(0) {}
    };
    
    std::unordered_map<std::string, std::unique_ptr<MetricData>> metrics_;
    size_t max_samples_;
    rclcpp::Logger logger_;
    
    // 计算标准差
    double calculateStdDev(const std::deque<double>& values, double mean) const {
        if (values.size() < 2) return 0.0;
        
        double sum_sq = 0.0;
        for (double v : values) {
            double diff = v - mean;
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / values.size());
    }
    
    // 计算平均值
    double calculateMean(const std::deque<double>& values) const {
        if (values.empty()) return 0.0;
        
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.size();
    }
    
public:
    PerformanceMonitor(size_t max_samples = 100, 
                      const rclcpp::Logger& logger = rclcpp::get_logger("PerformanceMonitor"))
        : max_samples_(max_samples), logger_(logger) {}
    
    // 开始计时
    void startTimer(const std::string& metric_name) {
        if (metrics_.find(metric_name) == metrics_.end()) {
            metrics_[metric_name] = std::make_unique<MetricData>(metric_name);
        }
        metrics_[metric_name]->start_time = std::chrono::steady_clock::now();
    }
    
    // 结束计时并记录
    void endTimer(const std::string& metric_name) {
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
        
        // 保持最大样本数
        while (it->second->values.size() > max_samples_) {
            it->second->values.pop_front();
        }
    }
    
    // 记录自定义值
    void recordValue(const std::string& metric_name, double value) {
        if (metrics_.find(metric_name) == metrics_.end()) {
            metrics_[metric_name] = std::make_unique<MetricData>(metric_name);
        }
        
        metrics_[metric_name]->values.push_back(value);
        metrics_[metric_name]->total_count++;
        
        while (metrics_[metric_name]->values.size() > max_samples_) {
            metrics_[metric_name]->values.pop_front();
        }
    }
    
    // 获取指标统计信息
    PerformanceMetrics getMetrics(const std::string& metric_name) const {
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
        
        return PerformanceMetrics{
            metric_name,
            mean,
            min_val,
            max_val,
            std_dev,
            data->total_count
        };
    }
    
    // 输出所有指标统计信息
    void reportAll() {
        RCLCPP_INFO(logger_, "========== 性能监控报告 ==========");
        
        for (const auto& [name, data] : metrics_) {
            if (data->values.empty()) {
                RCLCPP_INFO(logger_, "%s: 无数据 (总采样: %zu)", name.c_str(), data->total_count);
                continue;
            }
            
            auto metrics = getMetrics(name);
            RCLCPP_INFO(logger_, 
                       "%s: avg=%.2fms, min=%.2fms, max=%.2fms, std=%.2fms (采样: %zu)",
                       metrics.name.c_str(),
                       metrics.avg_ms,
                       metrics.min_ms,
                       metrics.max_ms,
                       metrics.std_dev_ms,
                       metrics.sample_count);
        }
        
        RCLCPP_INFO(logger_, "=================================");
    }
    
    // 清空所有指标
    void reset() {
        for (auto& [name, data] : metrics_) {
            data->values.clear();
            data->total_count = 0;
        }
    }
    
    // 获取所有指标名称
    std::vector<std::string> getMetricNames() const {
        std::vector<std::string> names;
        for (const auto& [name, data] : metrics_) {
            names.push_back(name);
        }
        return names;
    }
};

#endif // PERFORMANCE_MONITOR_H