/**
 * @file combat_manager.h
 * @brief 战斗管理器声明 —— 整合检测、跟踪、弹道补偿、射击指令发布的作战控制中心
 *
 * CombatManager 是自瞄系统的核心调度器，协调以下子系统：
 *   - YOLOv5Detector: 目标检测 + PnP 测距
 *   - TargetEKF: 目标状态滤波 + 位置预测
 *   - TrackingStateMachine: 跟踪状态管理（防误跟踪）
 *   - ShootController: 射击模式控制
 *   - PerformanceMonitor: 性能监控
 *
 * 两个定时器回调入口：
 *   shootLoop()  (30ms):  检测→滤波→跟踪状态机→弹道补偿→云台指向
 *   swingLoop()  (20ms):  摆动扫描→射击指令发布→开火计时
 *
 * 内部状态：
 *   - target_ekf_: 目标 EKF 滤波器实例
 *   - tracking_sm_: 跟踪状态机实例
 *   - last_ekf_*: 上一次 EKF 输出（用于 predict 外推）
 *   - detect_debug_count_: 检测帧计数（调试用）
 */

#ifndef COMPETITION_COMBAT_MANAGER_H
#define COMPETITION_COMBAT_MANAGER_H

#include <chrono>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "tdt_interface/msg/send_data.hpp"

#include "runtime_config.h"
#include "target_ekf.h"
#include "tracking_state_machine.h"

class YOLOv5Detector;
class PerformanceMonitor;
class PathNavigator;
class ShootController;
class Waypoint;
struct SharedContext;

class CombatManager {
public:
    CombatManager(SharedContext& ctx, const RuntimeConfig& config,
                  rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr angles_pub, YOLOv5Detector& detector,
                  PerformanceMonitor& perf_monitor, PathNavigator& path_navigator, ShootController& shoot_controller,
                  const std::vector<Waypoint>& waypoints, rclcpp::Logger logger);

    /** @brief 自瞄检测定时器回调（30ms / ~33Hz）—— 检测→滤波→弹道补偿 */
    void shootLoop();

    /** @brief 摆动+射击指令定时器回调（20ms / 50Hz）—— 摆动扫描→开火控制 */
    void swingLoop();

private:
    /** @brief 发射射击指令（设定云台指向和开火标志） */
    void publishShootCommand(size_t waypoint_idx, rclcpp::Time current_time);

    // ── 依赖引用 ──
    SharedContext& ctx_;                                                     ///< 共享上下文
    const RuntimeConfig& config_;                                            ///< 运行时配置
    rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr angles_pub_;  ///< 云台发布者
    YOLOv5Detector& detector_;                                               ///< 目标检测器
    PerformanceMonitor& perf_monitor_;                                       ///< 性能监控
    PathNavigator& path_navigator_;                                          ///< 路径导航
    ShootController& shoot_controller_;                                      ///< 射击控制
    const std::vector<Waypoint>& waypoints_;                                 ///< 路径点数组
    rclcpp::Logger logger_;                                                  ///< 日志器

    // ── 内部状态 ──
    TargetEKF target_ekf_;                                 ///< 目标 EKF 滤波器
    TrackingStateMachine tracking_sm_;                     ///< 跟踪状态机
    std::chrono::steady_clock::time_point last_ekf_time_;  ///< 上次 EKF 更新时间
    bool last_ekf_time_set_ = false;                       ///< EKF 时间是否已初始化
    int detect_debug_count_ = 0;                           ///< 检测帧计数（调试用）
    bool degradation_warned_ = false;                      ///< 是否已输出降级警告
};

#endif
