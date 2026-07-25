/**
 * @file path_navigator.h
 * @brief 路径点导航器声明 —— 管理当前目标、到点切换、卡住重试、超时跳过、回退
 *
 * PathNavigator 是 MovementController 的"导航大脑"，负责：
 *   1. 维护当前目标路径点索引
 *   2. 到达判定和路径点切换
 *   3. 卡住检测（重试计数 + 超时）
 *   4. 被攻击时回退到上一个路径点
 *   5. 强制跳转到指定路径点（自瞄攻击响应）
 *
 * 线程安全：所有方法加 mutex_ 保护
 *
 * 与 MovementController 的交互（5ms 定时器循环）：
 *   controlLoop → getCurrentWaypointIndex → distanceTo → isArrived
 *   → moveToNext / incrementRetryCount / skipCurrentPoint
 *
 * 路径索引结构：
 *   path_indices_[current_target_index_] → waypoints_ 的原始索引
 *   kLoopStart=6 控制路径循环起点（跳过起始段）
 */

#ifndef COMPETITION_PATH_NAVIGATOR_H
#define COMPETITION_PATH_NAVIGATOR_H

#include <memory>
#include <mutex>
#include <vector>

#include "rclcpp/rclcpp.hpp"

class IPathPlanningStrategy;

class PathNavigator {
    static constexpr size_t kLoopStart = 6;  ///< 路径循环起点（跳过 P1-P6 出发段）

private:
    int max_retry_count_;                 ///< 卡住重试次数上限
    std::vector<size_t> path_indices_;    ///< 路径点访问索引序列
    size_t current_target_index_;         ///< 当前目标在 path_indices_ 中的下标
    int stuck_retry_count_;               ///< 卡住重试计数
    rclcpp::Time start_time_to_current_;  ///< 当前目标的起始时间（用于超时判断）
    bool is_returning_;                   ///< 是否处于回退状态
    size_t return_to_index_;              ///< 回退目标索引
    bool max_retries_exceeded_;           ///< 重试次数是否已超限
    double move_timeout_;                 ///< 移动超时时间（秒）
    mutable std::mutex mutex_;            ///< 互斥锁
public:
    PathNavigator(const IPathPlanningStrategy& path_strategy, double move_timeout = 10.0, int max_retry_count = 3);
    ~PathNavigator();

    /** @brief 获取当前目标路径点在 waypoints_ 中的原始索引 */
    size_t getCurrentWaypointIndex() const;

    /** @brief 切换到下一个路径点（回退状态下跳到 return_to_index_） */
    void moveToNext(rclcpp::Time current_time);

    /** @brief 标记"下次 moveToNext 应回退到上一个点" */
    void markReturnToPrevious();

    /** @brief 取消回退状态 */
    void cancelReturn();

    /** @brief 强制导航到指定路径点（用于自瞄攻击响应） */
    void navigateToWaypoint(size_t waypoint_idx, rclcpp::Time current_time);

    /** @brief 跳过当前点，直接进入下一个 */
    void skipCurrentPoint(rclcpp::Time current_time);

    /** @brief 检查是否超时 */
    bool isTimeout(rclcpp::Time current_time);

    /** @brief 累加卡住重试计数（超过 MAX_RETRY_COUNT 触发跳过） */
    void incrementRetryCount(rclcpp::Time current_time);

    /** @brief 重置超时计时器 */
    void resetTimeoutTimer(rclcpp::Time current_time);

    /** @brief 重置卡住重试计数器 */
    void resetRetryCount();

    /** @brief 获取当前卡住重试次数 */
    int getRetryCount() const;

    /** @brief 检查重试次数是否已超限 */
    bool isMaxRetriesExceeded() const;

    /** @brief 检查是否处于回退状态 */
    bool isReturning() const;
};

#endif
