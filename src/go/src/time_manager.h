/**
 * @file time_manager.h
 * @brief 全局时间管理器声明 —— 纯静态类，维护游戏时间基准和时间工具函数
 *
 * TimeManager 是一个无实例的静态工具类，提供：
 *   - 游戏时间 ↔ ROS 时间的映射（getCurrentTime）
 *   - 超时判断（isTimeout）
 *   - 历史记录过期统计（countExpiredRecords）
 *   - 时间差计算（timeSince）
 *
 * 所有方法均为 static，可在任何地方直接调用（如 TimeManager::getCurrentTime()）。
 * 线程安全通过 atomic 变量保证（game_start_time_ 写一次后只读）。
 *
 * 初始化：GoNode 构造函数中调用 initialize(ros_time) 设置时间基准。
 * 更新：GoNode::gameTimeCallback() 每秒调用 updateGameTime() 同步游戏时间。
 */

#ifndef COMPETITION_TIME_MANAGER_H
#define COMPETITION_TIME_MANAGER_H

#include <deque>
#include <cstdint>
#include <atomic>
#include "rclcpp/rclcpp.hpp"

class TimeManager {
private:
    static rclcpp::Time game_start_time_;              ///< 程序启动时的 ROS 时间基准
    static std::atomic<int32_t> current_game_time_;    ///< 模拟器当前游戏时间（秒）
    static std::atomic<bool> initialized_;             ///< 是否已初始化

public:
    /** @brief 初始化时间基准（只执行一次） */
    static void initialize(rclcpp::Time ros_time);

    /** @brief 更新游戏时间（由 /game_time 回调调用，1Hz） */
    static void updateGameTime(int32_t game_time);

    /** @brief 获取当前时间的 ROS 时间表示（game_start + game_time） */
    static rclcpp::Time getCurrentTime();

    /** @brief 计算时间差（秒）：current_time - timestamp */
    static double timeSince(rclcpp::Time timestamp, rclcpp::Time current_time);

    /** @brief 判断是否超时：timeSince > timeout_seconds */
    static bool isTimeout(rclcpp::Time start_time, rclcpp::Time current_time, double timeout_seconds);

    /** @brief 统计时间戳队列中已过期的记录数（利用有序性提前退出） */
    static size_t countExpiredRecords(rclcpp::Time current_time,
                                       const std::deque<rclcpp::Time>& timestamps,
                                       double max_duration);
};

#endif
