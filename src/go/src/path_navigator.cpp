/**
 * @file path_navigator.cpp
 * @brief 路径点导航器 —— 管理当前目标路径点、到点判定、卡住重试、超时跳过、回退逻辑
 *
 * PathNavigator 是 MovementController 的"导航大脑"，负责决定机器人当前应该前往哪个路径点，
 * 以及到达后如何切换到下一个点。
 *
 * 核心职责：
 *   1. 维护当前目标路径点索引（在 PathStrategy 生成的路径序列中）
 *   2. 提供到达判定：distanceTo() + isArrived()，根据距离阈值分三档
 *   3. 卡住检测：如果某点停留时间过长或重试次数过多，自动跳过
 *   4. 超时跳过：移动超时后强制进入下一个路径点
 *   5. 回退支持：被攻击时可以回退到上一个路径点（markReturnToPrevious）
 *   6. 强制导航：navigateToWaypoint() 可跳转到任意路径点（用于自瞄攻击响应）
 *
 * 路径点索引结构：
 *   PathStrategy 生成的 path_indices_ 是一个索引数组，映射到 waypoints_ 中的实际坐标。
 *   例如 path_indices_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
 *   current_target_index_ 是 path_indices_ 中的下标（不是 waypoint 的原始索引）。
 *
 * 与 MovementController 的交互：
 *   MovementController 在 controlLoop() 中每5ms调用本模块的方法：
 *   - getCurrentWaypointIndex() → 获取当前目标点
 *   - distanceTo() → 计算到目标点的距离
 *   - isArrived() → 判断是否到达（根据距离阈值分三档）
 *   - isTimeout() → 检查是否超时
 *   - incrementRetryCount() → 每次到达判定失败时累加重试计数
 *   - moveToNext() → 到达后切换到下一个点
 *
 * 线程安全：所有方法加 mutex_ 保护，可在定时器线程和回调线程中安全调用
 */

#include "path_navigator.h"

#include "path_strategy.h"
#include "time_manager.h"

PathNavigator::~PathNavigator() = default;

/**
 * @brief 构造函数
 * @param path_strategy 已由 GoNode 从 YAML 路径点创建的策略
 * @param move_timeout 已验证的移动超时时间（秒）
 *
 * 初始化流程：
 *   1. 调用已注入策略的 generatePathIndices() 生成路径索引序列
 *   2. 保存移动超时和重试上限
 */
PathNavigator::PathNavigator(const IPathPlanningStrategy& path_strategy, double move_timeout, int max_retry_count)
    : max_retry_count_(max_retry_count),
      current_target_index_(0),
      stuck_retry_count_(0),
      is_returning_(false),
      return_to_index_(0),
      max_retries_exceeded_(false),
      move_timeout_(move_timeout) {
    path_indices_ = path_strategy.generatePathIndices();
}

/**
 * @brief 获取当前目标路径点在 waypoints_ 数组中的原始索引
 * @return 路径点索引，越界返回 0
 *
 * 注意：返回的是 waypoints_ 的索引，不是 path_indices_ 的索引。
 * path_indices_[current_target_index_] → waypoints_ 的索引
 */
size_t PathNavigator::getCurrentWaypointIndex() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (current_target_index_ < path_indices_.size()) {
        return path_indices_[current_target_index_];
    }
    return 0;
}

/**
 * @brief 切换到下一个路径点
 * @param current_time 当前时间（用于重置超时计时器）
 *
 * 切换逻辑：
 *   - 如果正在回退状态（is_returning_）→ 直接跳到 return_to_index_，不递增
 *   - 正常情况 → current_target_index_++，到达末尾则循环回 kLoopStart
 *   - 重置卡住重试计数器和超时计时器
 *
 * 循环起点 kLoopStart 通常 > 0，因为前几个路径点是出发点（不重复经过）
 */
void PathNavigator::moveToNext(rclcpp::Time current_time) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 回退模式：直接跳到之前记录的位置，不递增
    if (is_returning_) {
        current_target_index_ = return_to_index_;
        is_returning_ = false;
        return_to_index_ = 0;
        RCLCPP_DEBUG(rclcpp::get_logger("PathNavigator"), "返回到索引%zu", current_target_index_);
        return;
    }

    current_target_index_++;

    // 路径循环：到达末尾后回到 kLoopStart（跳过起始段的纯移动点）
    if (current_target_index_ >= path_indices_.size()) {
        current_target_index_ = kLoopStart;
    }

    stuck_retry_count_ = 0;
    max_retries_exceeded_ = false;
    start_time_to_current_ = current_time;
}

/**
 * @brief 标记"下一次 moveToNext 应回退到上一个路径点"
 *
 * 用于被攻击响应：StateManager 检测到被攻击时调用此方法，
 * 下次 movement_ 的 controlLoop 调用 moveToNext() 时会回退而非前进。
 *
 * 注意：只是设置标志，实际回退在 moveToNext() 中执行
 */
void PathNavigator::markReturnToPrevious() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (current_target_index_ > 0) {
        return_to_index_ = current_target_index_ - 1;
        is_returning_ = true;
    }
}

/**
 * @brief 强制导航到指定路径点（用于自瞄攻击响应）
 * @param waypoint_idx 目标路径点在 waypoints_ 中的原始索引
 * @param current_time 当前时间（用于重置超时计时器）
 *
 * 遍历 path_indices_ 找到匹配的索引，直接设置 current_target_index_。
 * 如果找不到目标点，则静默返回（不修改当前状态）。
 */
void PathNavigator::navigateToWaypoint(size_t waypoint_idx, rclcpp::Time current_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < path_indices_.size(); i++) {
        if (path_indices_[i] == waypoint_idx) {
            current_target_index_ = i;
            stuck_retry_count_ = 0;
            max_retries_exceeded_ = false;
            is_returning_ = false;
            return_to_index_ = 0;
            start_time_to_current_ = current_time;
            return;
        }
    }
}

/**
 * @brief 取消回退状态（用于自瞄攻击响应时打断回退）
 */
void PathNavigator::cancelReturn() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_returning_ = false;
    return_to_index_ = 0;
    RCLCPP_DEBUG(rclcpp::get_logger("PathNavigator"), "取消返回状态");
}

/**
 * @brief 跳过当前路径点，直接进入下一个
 * @param current_time 当前时间
 *
 * 用于卡住重试次数超限或移动超时的情况：
 *   1. 重置卡住计数器
 *   2. 调用 moveToNext() 切换到下一个点
 */
void PathNavigator::skipCurrentPoint(rclcpp::Time current_time) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stuck_retry_count_ = 0;
        max_retries_exceeded_ = false;
    }
    moveToNext(current_time);
}

/**
 * @brief 检查到达当前路径点的移动是否超时
 * @param current_time 当前时间
 * @return true=已超时，应跳过当前点
 *
 * 首次调用时会初始化 start_time_to_current_ 为当前时间。
 * 超时时间从配置读取（默认10秒），每次 moveToNext() 会重置计时器。
 */
bool PathNavigator::isTimeout(rclcpp::Time current_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (start_time_to_current_.nanoseconds() == 0) {
        start_time_to_current_ = current_time;
    }
    return TimeManager::isTimeout(start_time_to_current_, current_time, move_timeout_);
}

/**
 * @brief 累加卡住重试计数
 * @param current_time 当前时间（同时重置超时计时器）
 *
 * MovementController 每次 controlLoop() 中，如果机器人既没到达也没超时，
 * 会调用此方法累加一次重试计数。当计数超过 MAX_RETRY_COUNT 时，
 * isMaxRetriesExceeded() 返回 true，触发跳过逻辑。
 */
void PathNavigator::incrementRetryCount(rclcpp::Time current_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    stuck_retry_count_++;
    if (stuck_retry_count_ >= max_retry_count_) {
        max_retries_exceeded_ = true;
    }
    start_time_to_current_ = current_time;
}

/** @brief 重置超时计时器（在机器人有明显移动时调用） */
void PathNavigator::resetTimeoutTimer(rclcpp::Time current_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    start_time_to_current_ = current_time;
}

/** @brief 重置卡住重试计数器和超限标志 */
void PathNavigator::resetRetryCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    stuck_retry_count_ = 0;
    max_retries_exceeded_ = false;
}

/** @brief 获取当前卡住重试次数 */
int PathNavigator::getRetryCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stuck_retry_count_;
}

/** @brief 检查卡住重试次数是否已超限 */
bool PathNavigator::isMaxRetriesExceeded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_retries_exceeded_;
}

/** @brief 检查是否处于回退状态 */
bool PathNavigator::isReturning() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_returning_;
}
