/**
 * @file time_manager.cpp
 * @brief 全局时间管理器 —— 维护游戏时间基准，提供超时判断和历史记录裁剪
 *
 * TimeManager 是一个纯静态类（无实例），所有方法和数据都是 static。
 * 它维护两个时间概念：
 *
 * 1. ROS 系统时间（ros_time）：
 *    - 基于 steady_clock，单调递增，不受系统时间修改影响
 *    - 用于所有内部计时（超时判断、历史记录裁剪等）
 *    - game_start_time_ 记录了程序启动时的 ROS 时间
 *
 * 2. 游戏时间（game_time）：
 *    - 由模拟器通过 /game_time 话题发布，从0开始每秒递增
 *    - 用于 display_duration 等显示逻辑（模拟器倒计时）
 *    - 通过 getCurrentTime() 映射为 ROS 时间：game_start_time_ + game_time
 *
 * 为什么需要两个时间？
 *   - 游戏时间可能暂停/跳变（模拟器控制），不能用于内部计时
 *   - ROS 时间稳定单调，适合做超时判断和滑动窗口
 *   - getCurrentTime() 将游戏时间映射到 ROS 时间轴，兼顾两者
 *
 * 线程安全：
 *   - game_start_time_：只在 initialize() 中写入一次，后续只读（无锁安全）
 *   - current_game_time_：atomic<int32_t>，多线程无锁读写
 *   - initialized_：atomic<bool>，保证 initialize() 只执行一次（double-checked locking）
 *
 * 使用者：
 *   - DataManager：trimHistory() 调用 countExpiredRecords() 裁剪过期数据
 *   - PathNavigator：isTimeout() 调用 isTimeout() 判断移动超时
 *   - GoNode：healthCheckLoop() 调用 getCurrentTime() 获取当前时间戳
 */

#include "time_manager.h"

// ── 静态成员变量定义 ──

/** @brief 程序启动时的 ROS 时间基准（initialize 时设置） */
rclcpp::Time TimeManager::game_start_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

/** @brief 模拟器发布的当前游戏时间（秒，从0开始递增） */
std::atomic<int32_t> TimeManager::current_game_time_(0);

/** @brief 是否已初始化（防止重复初始化） */
std::atomic<bool> TimeManager::initialized_(false);

/**
 * @brief 初始化时间管理器（只执行一次）
 * @param ros_time 程序启动时的 ROS 时间（通常在 GoNode 构造函数中调用）
 *
 * 使用 acquire/release 语义保证 double-checked locking 的正确性：
 *   - acquire: 确保看到 store(true) release 之前的所有写入
 *   - release: 确保 store 之前的 game_start_time_ 赋值对其他线程可见
 */
void TimeManager::initialize(rclcpp::Time ros_time) {
    if (!initialized_.load(std::memory_order_acquire)) {
        game_start_time_ = ros_time;
        current_game_time_.store(0, std::memory_order_relaxed);
        initialized_.store(true, std::memory_order_release);
    }
}

/**
 * @brief 更新游戏时间（由 GoNode::gameTimeCallback 调用，1Hz）
 * @param game_time 模拟器发布的当前游戏时间（秒）
 */
void TimeManager::updateGameTime(int32_t game_time) {
    current_game_time_.store(game_time, std::memory_order_relaxed);
}

/**
 * @brief 获取当前时间的 ROS 时间表示
 * @return game_start_time_ + current_game_time_（映射到 ROS 时间轴）
 *
 * 用途：所有需要时间戳的操作（DataManager 记录、超时判断等）都使用此方法。
 * 返回值是 rclcpp::Time，可以直接与 ROS 时间戳做差计算。
 */
rclcpp::Time TimeManager::getCurrentTime() {
    return game_start_time_ + rclcpp::Duration::from_seconds(
        current_game_time_.load(std::memory_order_relaxed));
}

/**
 * @brief 计算两个时间戳之间的时间差（秒）
 * @param timestamp   起始时间（如历史记录的时间戳）
 * @param current_time 当前时间
 * @return 时间差（秒），current_time - timestamp
 *
 * 注意：rclcpp::Time 的减法返回 Duration，调用 .seconds() 转为 double
 */
double TimeManager::timeSince(rclcpp::Time timestamp, rclcpp::Time current_time) {
    return (current_time - timestamp).seconds();
}

/**
 * @brief 判断是否超时
 * @param start_time      超时起始时间
 * @param current_time    当前时间
 * @param timeout_seconds 超时阈值（秒）
 * @return true=已超时（时间差 > 阈值）
 *
 * 注意：使用严格大于（>）而非大于等于（>=），
 * 保证刚好等于阈值时不算超时（边界友好）
 */
bool TimeManager::isTimeout(rclcpp::Time start_time, rclcpp::Time current_time, double timeout_seconds) {
    return timeSince(start_time, current_time) > timeout_seconds;
}

/**
 * @brief 统计时间戳队列中有多少条记录已过期
 * @param current_time 当前时间
 * @param timestamps   按时间递增排列的时间戳队列
 * @param max_duration 最大保留时长（秒）
 * @return 已过期的记录数量
 *
 * 假设：timestamps 按时间单调递增（最早→最晚排列）
 * 遍历到第一个未过期的记录即停止（利用有序性提前退出）
 *
 * 至少保留1条记录：如果全部过期也不返回 timestamps.size()，
 * 而是返回 size()-1，保证队列不被清空。
 *
 * 使用者：DataManager 的 trimHistory() 模板函数调用此方法
 */
size_t TimeManager::countExpiredRecords(rclcpp::Time current_time,
                                         const std::deque<rclcpp::Time>& timestamps,
                                         double max_duration) {
    if (timestamps.size() <= 1) return 0;
    size_t expire_count = 0;
    for (const auto& ts : timestamps) {
        if (timeSince(ts, current_time) > max_duration) expire_count++;
        else break;  // 利用有序性：遇到第一个未过期的就停止
    }
    return expire_count;
}
