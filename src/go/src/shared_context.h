/**
 * @file shared_context.h
 * @brief 跨模块共享上下文 —— 存储机器人全局状态，供多个模块读写
 *
 * SharedContext 是 MovementController、CombatManager、StateManager 之间的"共享内存"，
 * 通过 mutex 保护并发访问，解决模块间的状态同步问题。
 *
 * 设计目的：
 *   避免模块之间直接互相引用（减少耦合），通过共享上下文间接通信。
 *   例如：CombatManager 检测到目标后写入 target_yaw/pitch，
 *   MovementController 读取这些值来决定云台指向。
 *
 * 包含三组状态：
 *   1. 无锁区域：player_id（init-only）, target_yaw/pitch（atomic）, is_precise_adjusting/target_locked（atomic）
 *   2. WithMutex<ChassisLocked>：state, is_under_attack, attack_waypoint_stack（持 chassis.mutex 访问）
 *   3. WithMutex<GimbalLocked>：current_yaw/pitch, last_detection_time, shooting_start_time（持 gimbal.mutex 访问）
 *   4. 检测状态（atomic，安全跨线程）：检测失败计数, 距离门控
 *
 * 状态机：
 *   chassis.state 在 MOVING 和 SHOOTING 之间切换：
 *   - MOVING:  正常导航模式，MovementController 控制底盘移动
 *   - SHOOTING: 射击模式，底盘减速/停止，CombatManager 控制云台指向
 *
 * 攻击响应机制：
 *   chassis.attack_waypoint_stack 是一个栈，存储被攻击时需要回退的路径点。
 *   被攻击时 push 当前点，响应完成后 pop 回到之前的位置。
 */

#ifndef COMPETITION_SHARED_CONTEXT_H
#define COMPETITION_SHARED_CONTEXT_H

#include <stack>
#include <mutex>
#include <atomic>
#include <chrono>
#include <opencv2/core.hpp>
#include "rclcpp/rclcpp.hpp"

/**
 * @brief 将数据和其保护锁打包为一个单元，使锁的作用域在类型层面可见
 *
 * 用法：
 *   auto lock = std::lock_guard(ctx_.chassis.mutex);
 *   ctx_.chassis.value.state = SHOOTING;
 */
template<typename T>
struct WithMutex {
    mutable std::mutex mutex;
    T value;
};

struct SharedContext {
    /** @brief 底盘工作状态枚举 */
    enum State { MOVING, SHOOTING };

    // ═══════════════════════════════════════════════
    // 无锁区域（初始化不变 / atomic / 单写者）
    // ═══════════════════════════════════════════════

    int player_id = 0;                  ///< 玩家ID（1=蓝方，2=红方，初始化后不变）
    std::atomic<double> target_yaw{0};  ///< 当前目标云台 yaw（度）
    std::atomic<double> target_pitch{0};///< 当前目标云台 pitch（度）
    std::atomic<bool> is_precise_adjusting{false};  ///< 是否处于精调模式（atomic）
    std::atomic<bool> target_locked{false};         ///< 目标是否已锁定（atomic）

    // shootLoop 写入 + 其他线程可读（atomic 保证可见性）
    std::atomic<int> detection_fail_count{0};                  ///< 连续检测失败计数
    std::atomic<double> last_accepted_distance{0.0};           ///< 上次接受的测量距离
    std::atomic<bool> apply_distance_gate{false};              ///< 是否启用距离门控

    // 以下变量仅限 swingLoop 线程访问，无竞争
    std::chrono::steady_clock::time_point last_swing_time;     ///< 上次进入摆动模式的时间
    bool last_swing_time_set = false;                          ///< last_swing_time 是否已初始化

    // ═══════════════════════════════════════════════
    // 互斥锁保护区域（持锁后方可读写 .value）
    // ═══════════════════════════════════════════════

    struct ChassisLocked {
        State state = MOVING;                          ///< 当前工作状态
        bool is_under_attack = false;                  ///< 是否正在被攻击
        std::stack<size_t> attack_waypoint_stack;      ///< 被攻击时的路径点回退栈
    };
    WithMutex<ChassisLocked> chassis;

    struct GimbalLocked {
        double current_yaw = 0;                                    ///< 当前云台 yaw
        double current_pitch = 0;                                  ///< 当前云台 pitch
        rclcpp::Time last_detection_time{0, 0, RCL_ROS_TIME};    ///< 上次检测到目标的时间
        rclcpp::Time shooting_start_time{0, 0, RCL_ROS_TIME};    ///< 开始射击的时间
    };
    WithMutex<GimbalLocked> gimbal;
};

#endif
