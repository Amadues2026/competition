/**
 * @file state_manager.h
 * @brief 状态管理器声明 —— 血量监控、自瞄降级决策、被攻击响应、性能报告
 *
 * StateManager 是决策层的"大脑"，负责：
 *   1. 血量监控（healthCheckLoop，1Hz）：分析自身和敌方血量变化
 *   2. 自瞄降级：根据敌方血量变化量决定是否优先攻击某个目标
 *   3. 被攻击响应：检测到自身被攻击时回退路径点、调整战术
 *   4. 目标击杀处理：敌方被消灭后切换到下一个目标
 *   5. 性能报告：定期输出系统性能数据
 *
 * 核心决策逻辑：
 *   - 敌方血量变化量 > 阈值 → 说明被队友打中 → 优先集火
 *   - 自身血量过低（< 20%）→ 考虑撤退
 *   - 连续未命中过多 → 切换摆动扫描模式
 *   - 检测持续失败 → 禁用自瞄（可能目标已消失）
 */

#ifndef COMPETITION_STATE_MANAGER_H
#define COMPETITION_STATE_MANAGER_H

#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "tdt_interface/msg/send_data.hpp"

#include "runtime_config.h"
#include "waypoint_utils.h"

class DataManager;
class PathNavigator;
class ShootController;
class PerformanceMonitor;
class Waypoint;
struct SharedContext;

class StateManager {
public:
    /**
     * @brief 构造函数
     * @param ctx             共享上下文引用
     * @param shoot_controller 射击控制器引用
     * @param data_manager    数据管理器引用
     * @param path_navigator  路径导航器引用
     * @param perf_monitor    性能监控器指针（可为 nullptr）
     * @param health_thresholds 已验证的血量变化阈值
     * @param config          已验证的运行时配置
     * @param waypoints       路径点数组只读引用
     * @param angles_pub      云台角度发布者
     * @param logger          ROS2 日志器
     */
    StateManager(SharedContext& ctx, ShootController& shoot_controller, DataManager& data_manager,
                 PathNavigator& path_navigator, PerformanceMonitor* perf_monitor,
                 const waypoint_utils::HealthChangeThresholdConfig& health_thresholds, const RuntimeConfig& config,
                 const std::vector<Waypoint>& waypoints,
                 rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr angles_pub, rclcpp::Logger logger);

    /** @brief 血量监控主循环（1Hz）—— 分析血量变化，做出降级/集火决策 */
    void healthCheckLoop();

    /**
     * @brief 处理自身被攻击事件
     * @param waypoint_idx        当前路径点索引
     * @param current_enemy_health 当前敌方血量
     * @param initial_enemy_health 敌方初始血量
     */
    void handleSelfUnderAttack(size_t waypoint_idx, int current_enemy_health, int initial_enemy_health);

private:
    /** @brief 尝试输出性能报告（每 N 次检查输出一次） */
    void tryReportPerformance();

    /**
     * @brief 评估血量变化并执行自瞄降级决策
     * @param waypoint_idx  当前路径点索引（决定阈值和攻击目标）
     * @param current_time  当前时间
     */
    void evaluateHealthAndAutoAim(size_t waypoint_idx, rclcpp::Time current_time);

    /**
     * @brief 处理敌方单位被击杀
     * @param waypoint_idx  当前路径点索引
     * @param current_time  当前时间
     */
    void handleKilledEnemy(size_t waypoint_idx, rclcpp::Time current_time);

    // ── 指令发布 ──
    void publishAngleCommand(double yaw, double pitch, bool if_shoot);
    void moveToWaypoint(size_t waypoint_idx, rclcpp::Time current_time);

    // ── 依赖引用 ──
    SharedContext& ctx_;
    ShootController& shoot_controller_;
    DataManager& data_manager_;
    PathNavigator& path_navigator_;
    PerformanceMonitor* perf_monitor_;                               ///< 可为 nullptr
    waypoint_utils::HealthChangeThresholdConfig health_thresholds_;  ///< 血量变化阈值
    const RuntimeConfig& config_;                                    ///< 运行时配置
    const std::vector<Waypoint>& waypoints_;
    rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr angles_pub_;
    rclcpp::Logger logger_;
    int health_check_count_ = 0;  ///< 检查计数（用于性能报告频率控制）
};

#endif
