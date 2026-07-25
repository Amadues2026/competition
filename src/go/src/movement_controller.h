/**
 * @file movement_controller.h
 * @brief 运动控制器声明 —— 底盘速度控制、距离分档、精调、位置保持、射击漂移补偿
 *
 * MovementController 是底盘运动的决策中心，负责：
 *   1. 根据到目标点的距离分三档调速（远/近/精调）
 *   2. 到达判定和路径点切换
 *   3. 超时/卡住检测和跳过
 *   4. 射击期间的位置保持（补偿后坐力漂移）
 *   5. 精调模式（低速高精度停靠）
 *
 * 两个定时器回调入口：
 *   controlLoop()         (5ms):  距离计算→速度分档→到达判定→发布速度指令
 *   positionAdjustLoop()  (可配): 射击期间的位置微调
 *
 * 速度曲线：
 *   距离 > speed_switch_distance     → max_speed_far（全速奔跑）
 *   距离 > precise_mode_distance     → 线性插值 max_far → max_near
 *   距离 > arrival_threshold_normal  → max_near（近距离）
 *   距离 ≤ precise_arrival_threshold → precise_adjust_speed（精调）
 *   到达                            → 停止
 */

#ifndef COMPETITION_MOVEMENT_CONTROLLER_H
#define COMPETITION_MOVEMENT_CONTROLLER_H

#include <vector>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tdt_interface/msg/send_data.hpp"

#include "runtime_config.h"

class DataManager;
class PathNavigator;
class ShootController;
class Waypoint;
struct SharedContext;

class MovementController {
public:
    /**
     * @brief 构造函数
     * @param ctx             共享上下文引用
     * @param config          运行时配置引用
     * @param speed_pub       底盘速度发布者
     * @param angles_pub      云台角度发布者
     * @param data_manager    数据管理器引用
     * @param path_navigator  路径导航器引用
     * @param shoot_controller 射击控制器引用
     * @param waypoints       路径点数组只读引用
     * @param logger          ROS2 日志器
     */
    MovementController(SharedContext& ctx, const RuntimeConfig& config,
                       rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr speed_pub,
                       rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr angles_pub, DataManager& data_manager,
                       PathNavigator& path_navigator, ShootController& shoot_controller,
                       const std::vector<Waypoint>& waypoints, rclcpp::Logger logger);

    /** @brief 运动控制主循环（5ms）—— 距离计算→速度分档→到达判定→指令发布 */
    void controlLoop();

    /** @brief 位置保持回调（射击期间位置微调） */
    void positionAdjustLoop();

    /**
     * @brief 精调模式移动（低速高精度停靠）
     * @param dx, dy   到目标的位移
     * @param distance 到目标的距离
     */
    void preciseAdjustment(double dx, double dy, double distance);

    /**
     * @brief 强制导航到指定路径点
     * @param waypoint_idx  目标路径点索引
     * @param current_time  当前时间
     */
    void moveToWaypoint(size_t waypoint_idx, rclcpp::Time current_time);

private:
    // ── 内部决策方法 ──

    /** @brief 处理移动超时和卡住重试 */
    bool handleTimeoutRetry(size_t waypoint_idx, rclcpp::Time current_time);

    /** @brief 处理射击期间的漂移补偿 */
    bool handleShootingDrift(size_t waypoint_idx, double distance, rclcpp::Time current_time);

    /** @brief 尝试开始射击（到达射击点且满足条件） */
    bool tryStartShooting(size_t waypoint_idx, const Waypoint& target, rclcpp::Time current_time);

    /** @brief 到达后推进到下一个路径点 */
    void advancePastWaypoint(size_t waypoint_idx, rclcpp::Time current_time);

    /** @brief 根据距离分档计算速度并发布 */
    void moveTowardsTarget(double dx, double dy, double distance, const Waypoint& target, size_t waypoint_idx);

    // ── 指令发布 ──
    void publishVelocity(double vx, double vy, rclcpp::Time stamp);
    void publishAngleCommand(double yaw, double pitch, bool if_shoot);

    // ── 依赖引用 ──
    SharedContext& ctx_;
    const RuntimeConfig& config_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr speed_pub_;
    rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr angles_pub_;
    DataManager& data_manager_;
    PathNavigator& path_navigator_;
    ShootController& shoot_controller_;
    const std::vector<Waypoint>& waypoints_;
    rclcpp::Logger logger_;
};

#endif
