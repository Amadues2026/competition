/**
 * @file data_manager.cpp
 * @brief 机器人运行时数据管理中心 —— 存储并查询自身/敌方血量、位置、云台角度的历史记录
 *
 * DataManager 是整个系统的"数据仓库"，所有模块通过它读写运行时数据：
 *   - 自身血量 + 血量变化量（用于 StateManager 判断自瞄降级）
 *   - 6个敌方单位的血量 + 初始血量 + 血量变化量（用于自瞄目标选择）
 *   - 机器人位置历史（用于 MovementController 位置保持）
 *   - 云台 yaw/pitch 历史（用于历史角度回溯查询）
 *
 * 核心设计：
 *   - 所有数据以 deque 双端队列存储，带时间戳
 *   - 每次更新自动调用 trimHistory() 裁剪过期数据，保持内存恒定
 *   - 血量变化量计算逻辑：每隔 health_history_duration 秒对比当前与历史血量，
 *     差值存入 changes 队列，供外部查询最近一次变化量
 *   - 线程安全：所有公开方法加 mutex_ 保护，getSelfHealth() 使用 atomic 无锁快速读取
 *
 * 与其他模块的关系：
 *   GoNode 回调 → updateXxx() 写入数据
 *   StateManager → getSelfHealthChange() 读取自身血量变化
 *   CombatManager → getEnemyHealthChange() / getInitialEnemyHealth() 读取敌方数据
 *   MovementController → getCurrentPosition() 读取当前位置
 *   YOLOv5Detector → getHistoricalYawPitch() 读取历史云台角度（用于弹道补偿）
 */

#include "data_manager.h"
#include "time_manager.h"
#include <cmath>


/**
 * @brief 通用历史记录裁剪模板函数
 * @param values 数据队列（如血量、位置、角度等）
 * @param timestamps 对应的时间戳队列
 * @param current 当前时间
 * @param duration 保留时长（秒），超过此时间的数据会被删除
 *
 * 假设：timestamps 与 values 一一对应，且按时间单调递增
 * 裁剪后至少保留 1 条记录（即使过期也不清空）
 */
template<typename T>
static void trimHistory(std::deque<T>& values, std::deque<rclcpp::Time>& timestamps,
                        rclcpp::Time current, double duration) {
    size_t expire_count = TimeManager::countExpiredRecords(current, timestamps, duration);
    if (expire_count > 0) {
        values.erase(values.begin(), values.begin() + expire_count);
        timestamps.erase(timestamps.begin(), timestamps.begin() + expire_count);
    }
}

/**
 * @brief 构造函数 —— 初始化血量历史和6个敌方单位的初始状态
 * @param health_history_duration 血量历史保留时长（秒），默认5.0
 * @param other_data_duration 位置/角度历史保留时长（秒），默认2.0
 */
DataManager::DataManager(double health_history_duration, double other_data_duration,
                         int max_reasonable_health_change,
                         double health_history_tolerance,
                         int max_health_changes,
                         double yaw_pitch_lookback)
    : atomic_self_health_(0),
      health_history_duration_(health_history_duration),
      other_data_duration_(other_data_duration),
      max_reasonable_health_change_(max_reasonable_health_change),
      health_history_tolerance_(health_history_tolerance),
      max_health_changes_(max_health_changes),
      yaw_pitch_history_lookback_(yaw_pitch_lookback) {
    for (int i = 0; i < kNumEnemies; i++) {
        initial_enemy_health_[i] = 0;
        initial_enemy_health_set_[i] = false;
    }
}

/**
 * @brief 更新自身血量并计算血量变化量
 * @param health 当前血量
 * @param timestamp 当前时间戳（来自 ROS2 消息头）
 *
 * 血量变化量计算流程：
 *   1. 将新血量追加到 history 队列
 *   2. trimHistory() 裁剪过期数据
 *   3. 等待 history 跨越完整一个 health_history_duration 周期
 *   4. 取最早一条记录的血量作为 "N秒前的血量"
 *   5. change = 当前血量 - N秒前的血量
 *   6. 过滤异常值：|change| > MAX_REASONABLE_HEALTH_CHANGE 则丢弃
 *   7. 追加到 changes 队列（最多保留 MAX_HEALTH_CHANGES 条）
 *
 * 注意：atomic_self_health_ 使用 memory_order_relaxed 快速写入，
 * 供 getSelfHealth() 无锁读取（高频调用场景）
 */
void DataManager::updateSelfHealth(int health, rclcpp::Time timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    atomic_self_health_.store(health, std::memory_order_relaxed);
    self_health_history_.push_back(health);
    self_health_timestamps_.push_back(timestamp);

    trimHistory(self_health_history_, self_health_timestamps_, timestamp, health_history_duration_);
    if (self_health_history_.size() <= 1) return;

    double time_to_oldest = TimeManager::timeSince(self_health_timestamps_.front(), timestamp);
    if (time_to_oldest < health_history_duration_ - health_history_tolerance_) return;

    int health_history_front = self_health_history_.front();
    int change = health - health_history_front;
    if (std::abs(change) > max_reasonable_health_change_) return;

    self_health_changes_.push_back(change);
    if (self_health_changes_.size() > max_health_changes_) {
        self_health_changes_.pop_front();
    }
}

/**
 * @brief 更新敌方单位血量并计算其血量变化量
 * @param enemy_index 敌方内部索引（0-5，非血量话题索引）
 * @param health 该敌方单位当前血量
 * @param timestamp 时间戳
 *
 * 流程与 updateSelfHealth() 相同，额外操作：
 *   - 首次收到某敌方血量时，记录为 initial_enemy_health_
 *   - 初始血量用于 StateManager 的自瞄降级判断
 */
void DataManager::updateEnemyHealth(int enemy_index, int health, rclcpp::Time timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enemy_index < 0 || enemy_index >= kNumEnemies) return;
    if (!initial_enemy_health_set_[enemy_index]) {
        initial_enemy_health_[enemy_index] = health;
        initial_enemy_health_set_[enemy_index] = true;
    }
    enemy_health_history_[enemy_index].push_back(health);
    enemy_health_timestamps_[enemy_index].push_back(timestamp);

    trimHistory(enemy_health_history_[enemy_index], enemy_health_timestamps_[enemy_index],
                timestamp, health_history_duration_);
    if (enemy_health_history_[enemy_index].size() <= 1) return;

    double time_to_oldest = TimeManager::timeSince(enemy_health_timestamps_[enemy_index].front(), timestamp);
    if (time_to_oldest < health_history_duration_ - health_history_tolerance_) return;

    int health_history_front = enemy_health_history_[enemy_index].front();
    int change = health - health_history_front;
    if (std::abs(change) > max_reasonable_health_change_) return;

    enemy_health_changes_[enemy_index].push_back(change);
    if (enemy_health_changes_[enemy_index].size() > max_health_changes_) {
        enemy_health_changes_[enemy_index].pop_front();
    }
}

/**
 * @brief 更新机器人位置（世界坐标系 x, y）
 * @param x 世界坐标 X（单位：地图坐标，范围 [-50, 50]）
 * @param y 世界坐标 Y（单位：地图坐标，范围 [-50, 50]）
 * @param timestamp 时间戳
 */
void DataManager::updatePosition(double x, double y, rclcpp::Time timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    position_history_.push_back({x, y, timestamp});
    position_timestamps_.push_back(timestamp);
    trimHistory(position_history_, position_timestamps_, timestamp, other_data_duration_);
}

/**
 * @brief 更新云台 yaw/pitch 历史
 * @param yaw 水平角度（度，0-360循环）
 * @param pitch 俯仰角度（度，正数为上仰）
 * @param timestamp 时间戳
 */
void DataManager::updateYawPitch(double yaw, double pitch, rclcpp::Time timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    yaw_pitch_history_.push_back({yaw, pitch, timestamp});
    yaw_pitch_timestamps_.push_back(timestamp);
    trimHistory(yaw_pitch_history_, yaw_pitch_timestamps_, timestamp, other_data_duration_);
}

/** @brief 获取自身血量（无锁快速读取，供高频定时器使用） */
int DataManager::getSelfHealth() const {
    return atomic_self_health_.load(std::memory_order_relaxed);
}

/** @brief 获取自身最近一次血量变化量（单位：HP） */
int DataManager::getSelfHealthChange() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (self_health_changes_.empty()) return 0;
    return self_health_changes_.back();
}

/**
 * @brief 获取敌方单位最近一次血量变化量
 * @param enemy_index 敌方内部索引（0-5）
 * @return 血量变化量（HP），越界或无数据返回 0
 */
int DataManager::getEnemyHealthChange(int enemy_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enemy_index < 0 || enemy_index >= kNumEnemies) return 0;
    if (enemy_health_changes_[enemy_index].empty()) return 0;
    return enemy_health_changes_[enemy_index].back();
}

/**
 * @brief 获取敌方单位的初始血量（首次收到该单位血量时的值）
 * @param enemy_index 敌方内部索引（0-5）
 * @return 初始血量，未初始化返回 0
 */
int DataManager::getInitialEnemyHealth(int enemy_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enemy_index < 0 || enemy_index >= kNumEnemies) return 0;
    return initial_enemy_health_[enemy_index];
}

/** @brief 检查敌方单位的初始血量是否已记录 */
bool DataManager::isEnemyHealthInitialized(int enemy_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enemy_index < 0 || enemy_index >= kNumEnemies) return false;
    return initial_enemy_health_set_[enemy_index];
}

/**
 * @brief 获取敌方单位当前血量
 * @param enemy_index 敌方内部索引（0-5）
 * @return 最新血量，无数据返回 0
 */
int DataManager::getCurrentEnemyHealth(int enemy_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enemy_index < 0 || enemy_index >= kNumEnemies) return 0;
    if (enemy_health_history_[enemy_index].empty()) return 0;
    return enemy_health_history_[enemy_index].back();
}


/**
 * @brief 获取历史时刻的云台 yaw/pitch（用于弹道补偿）
 * @param current_time 当前时间
 * @return {yaw, pitch} 元组
 *
 * 查找逻辑：从最新数据往前遍历，找到距当前时间恰好超过 kYawPitchHistoryLookback 的记录。
 * 如果找不到（历史太短），返回最近一条记录。
 * 如果没有任何数据，返回 {0, 0}。
 *
 * 用途：YOLOv5Detector 在计算弹道补偿时，需要知道"云台在弹丸飞行期间的预计指向"，
 * 而非当前瞬间的指向。通过回溯历史角度，可以更好地预测弹丸飞行中的弹道。
 */
std::tuple<double, double> DataManager::getHistoricalYawPitch(rclcpp::Time current_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = yaw_pitch_history_.rbegin(); it != yaw_pitch_history_.rend(); ++it) {
        if (TimeManager::timeSince(std::get<2>(*it), current_time) >= yaw_pitch_history_lookback_) {
            return {std::get<0>(*it), std::get<1>(*it)};
        }
    }
    if (!yaw_pitch_history_.empty()) {
        return {std::get<0>(yaw_pitch_history_.back()), std::get<1>(yaw_pitch_history_.back())};
    }
    return ZERO_TUPLE;
}

/** @brief 获取机器人当前位置 (x, y)，无数据返回 {0, 0} */
std::tuple<double, double> DataManager::getCurrentPosition() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (position_history_.empty()) return ZERO_TUPLE;
    return {std::get<0>(position_history_.back()), std::get<1>(position_history_.back())};
}
