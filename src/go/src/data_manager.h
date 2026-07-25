/**
 * @file data_manager.h
 * @brief 数据管理中心声明 —— 存储自身/敌方血量、位置、云台角度的历史记录
 *
 * DataManager 是整个系统的"数据仓库"，所有模块通过它读写运行时数据。
 *
 * 存储内容：
 *   - 自身血量 + 血量变化量（供 StateManager 自瞄降级判断）
 *   - 6个敌方单位的血量 + 初始血量 + 血量变化量（供目标选择）
 *   - 机器人位置历史（供 MovementController 位置保持）
 *   - 云台 yaw/pitch 历史（供弹道补偿角度回溯）
 *
 * 线程安全：
 *   - getSelfHealth(): atomic 无锁快速读取（200Hz 调用场景）
 *   - 其他所有方法: mutex_ 保护
 *
 * 内存管理：
 *   所有数据以 deque 存储，每次更新自动裁剪过期数据（由 TimeManager::countExpiredRecords 驱动）
 *
 * 常量：
 *   kNumEnemies=6, MAX_REASONABLE_HEALTH_CHANGE=500, MAX_HEALTH_CHANGES=10
 */

#ifndef COMPETITION_DATA_MANAGER_H
#define COMPETITION_DATA_MANAGER_H

#include <atomic>
#include <deque>
#include <tuple>
#include <mutex>
#include "rclcpp/rclcpp.hpp"
#include "waypoint_utils.h"

class DataManager {
    static constexpr int kNumEnemies = waypoint_utils::kNumEnemies;
    static inline const std::tuple<double, double> ZERO_TUPLE = {0.0, 0.0};

private:
    // ── 内部常量（从配置文件加载） ──
    int max_reasonable_health_change_;       ///< 单次血量变化上限
    double health_history_tolerance_;        ///< 血量历史窗口容差（秒）
    int max_health_changes_;                 ///< 血量变化记录最大条数
    double yaw_pitch_history_lookback_;      ///< 云台角度回溯时间（秒）

    // ── 自身血量数据 ──
    std::atomic<int> atomic_self_health_;                   ///< 当前血量（无锁快速读取）
    std::deque<int> self_health_history_;                   ///< 血量历史
    std::deque<rclcpp::Time> self_health_timestamps_;      ///< 对应时间戳
    std::deque<int> self_health_changes_;                   ///< 血量变化量记录

    // ── 敌方血量数据（每个敌方单位独立存储） ──
    std::deque<int> enemy_health_history_[kNumEnemies];     ///< 各敌方血量历史
    std::deque<rclcpp::Time> enemy_health_timestamps_[kNumEnemies];  ///< 对应时间戳
    int initial_enemy_health_[kNumEnemies];                 ///< 初始血量（首次收到时记录）
    bool initial_enemy_health_set_[kNumEnemies];            ///< 初始血量是否已记录
    std::deque<int> enemy_health_changes_[kNumEnemies];     ///< 各敌方血量变化量

    // ── 位置和角度数据 ──
    std::deque<std::tuple<double, double, rclcpp::Time>> position_history_;      ///< (x, y, time)
    std::deque<rclcpp::Time> position_timestamps_;                              ///< 位置时间戳
    std::deque<std::tuple<double, double, rclcpp::Time>> yaw_pitch_history_;    ///< (yaw, pitch, time)
    std::deque<rclcpp::Time> yaw_pitch_timestamps_;                             ///< 角度时间戳

    mutable std::mutex mutex_;               ///< 互斥锁（保护所有非 atomic 数据）
    double health_history_duration_;         ///< 血量历史保留时长（秒）
    double other_data_duration_;             ///< 位置/角度历史保留时长（秒）

public:
    DataManager(double health_history_duration = 5.0,
                double other_data_duration = 10.0,
                int max_reasonable_health_change = 500,
                double health_history_tolerance = 0.1,
                int max_health_changes = 10,
                double yaw_pitch_lookback = 1.0);

    // ── 写入方法（由 GoNode 回调调用） ──

    /** @brief 更新自身血量并计算血量变化量 */
    void updateSelfHealth(int health, rclcpp::Time timestamp);

    /** @brief 更新敌方单位血量（内部索引 0-5） */
    void updateEnemyHealth(int enemy_index, int health, rclcpp::Time timestamp);

    /** @brief 更新机器人位置 (x, y) */
    void updatePosition(double x, double y, rclcpp::Time timestamp);

    /** @brief 更新云台 yaw/pitch */
    void updateYawPitch(double yaw, double pitch, rclcpp::Time timestamp);

    // ── 读取方法（由各模块调用） ──

    /** @brief 获取自身血量（无锁，200Hz 安全） */
    int getSelfHealth() const;

    /** @brief 获取自身最近一次血量变化量 */
    int getSelfHealthChange() const;

    /** @brief 获取敌方单位最近一次血量变化量 */
    int getEnemyHealthChange(int enemy_index) const;

    /** @brief 获取敌方单位初始血量（首次收到时记录的值） */
    int getInitialEnemyHealth(int enemy_index) const;

    /** @brief 检查敌方初始血量是否已记录 */
    bool isEnemyHealthInitialized(int enemy_index) const;

    /** @brief 获取敌方单位当前血量 */
    int getCurrentEnemyHealth(int enemy_index) const;

    /**
     * @brief 获取历史时刻的云台 yaw/pitch（回溯 kYawPitchHistoryLookback 秒）
     * 用途：弹道补偿需要知道弹丸飞行期间的预计云台指向
     */
    std::tuple<double, double> getHistoricalYawPitch(rclcpp::Time current_time);

    /** @brief 获取机器人当前位置 (x, y) */
    std::tuple<double, double> getCurrentPosition() const;
};

#endif
