/**
 * @file shoot_controller.h
 * @brief 射击控制器声明 —— 射击模式状态机、自瞄开关、摆动扫描、连续未命中检测
 *
 * ShootController 管理射击相关的核心状态：
 *   - AimMode: 自瞄模式（AUTO/HARDCODE/SWING）
 *   - is_shooting_: 是否正在射击
 *   - 连续未命中计数（用于自动切换摆动/禁用自瞄）
 *
 * 三种自瞄模式：
 *   AUTO    - 正常自瞄（使用 YOLOv5 检测 + 弹道补偿）
 *   HARDCODE - 硬编码模式（固定云台角度，调试用）
 *   SWING   - 摆动扫描模式（左右摆动搜索目标）
 *
 * 线程安全：所有状态修改方法加 mutex_ 保护
 *
 * 与其他模块的交互：
 *   - MovementController: isShooting() 判断是否进入射击状态
 *   - CombatManager: startShooting/stopSwinging 控制模式切换
 *   - StateManager: shouldDisableAutoAim/shouldStartSwinging 检查降级条件
 */

#ifndef COMPETITION_SHOOT_CONTROLLER_H
#define COMPETITION_SHOOT_CONTROLLER_H

#include <mutex>

#include "rclcpp/rclcpp.hpp"

struct ShootControllerConfig {
    double swing_range = 6.0;
    double swing_speed = 10.0;
    int consecutive_miss_disable = 10;
    int consecutive_miss_swing = 3;
};

class ShootController {
public:
    /** @brief 自瞄模式枚举 */
    enum class AimMode { AUTO, HARDCODE, SWING };

private:
    AimMode aim_mode_;                        ///< 当前自瞄模式
    bool is_shooting_;                        ///< 是否正在射击
    int consecutive_miss_count_for_disable_;  ///< 连续未命中计数（禁用判定用）
    int consecutive_miss_count_for_swing_;    ///< 连续未命中计数（摆动判定用）
    double swing_offset_yaw_;                 ///< 相对路径点预设 yaw 的摆动偏移
    double swing_direction_;                  ///< 摆动方向（+1/-1）
    rclcpp::Logger logger_;                   ///< ROS2 日志器
    mutable std::mutex mutex_;                ///< 互斥锁

    // ── 从配置加载的参数 ──
    double swing_range_;                      ///< 摆动角度范围（度）
    double swing_speed_;                      ///< 摆动角速度（度/秒）
    int consecutive_miss_threshold_disable_;  ///< 禁用自瞄的连续未命中阈值
    int consecutive_miss_threshold_swing_;    ///< 切换摆动的连续未命中阈值

public:
    /** @brief 构造函数（接收已验证的摆动和阈值参数） */
    explicit ShootController(const ShootControllerConfig& config);

    // ── 射击状态控制 ──

    /** @brief 开始射击（记录初始血量和开始时间） */
    void startShooting(int initial_health, rclcpp::Time current_time);

    /** @brief 停止射击 */
    void stopShooting();

    /** @brief 禁用自瞄（连续未命中超限时调用） */
    void disableAutoAim();

    /** @brief 查询是否正在射击 */
    bool isShooting() const;

    /** @brief 查询自瞄是否启用（模式为 AUTO） */
    bool isAutoAimEnabled() const;

    // ── 未命中检测 ──

    /**
     * @brief 检查血量变化是否满足射击效果
     * @param health_change 敌方血量变化量
     * @param threshold     阈值
     * @param current_time  当前时间
     * @return true=命中（血量变化超过阈值）
     */
    bool checkHealthChange(int health_change, int threshold, rclcpp::Time current_time);

    /** @brief 检查是否应禁用自瞄（连续未命中超限） */
    bool shouldDisableAutoAim(rclcpp::Time);

    /** @brief 检查是否应切换到摆动扫描模式 */
    bool shouldStartSwinging(rclcpp::Time);

    /** @brief 获取摆动判定用的连续未命中计数 */
    int getConsecutiveMissCountForSwing() const;

    // ── 摆动扫描控制 ──

    /** @brief 开始摆动扫描（偏移从 0 开始，在 YAML 配置范围内往返） */
    void startSwinging(rclcpp::Time current_time);

    /** @brief 停止摆动扫描 */
    void stopSwinging(rclcpp::Time current_time);

    /**
     * @brief 更新三角波摆动 yaw 偏移量
     * @param dt 时间间隔（秒）
     * @return 相对路径点预设 yaw 的偏移量（范围 [-swing_range, +swing_range]）
     */
    double updateSwingYaw(double dt);

    /** @brief 查询是否正在摆动 */
    bool isSwinging() const;
};

#endif
