/**
 * @file movement_controller.cpp
 * @brief 运动控制器 —— 底盘移动、到位判定、状态切换
 *
 * 本模块运行两个定时循环：
 *
 * 1. controlLoop()（5ms）：主控制循环
 *    - MOVING 状态：向目标路径点移动，四段变速控制
 *    - 到位后：精确微调 → 进入 SHOOTING 或跳过（无射击任务的路径点）
 *    - SHOOTING 状态：位置漂移补偿（保持在射击点附近）
 *
 * 2. positionAdjustLoop()（可配置间隔）：射击期间位置保持
 *    - 仅在 SHOOTING 状态下工作
 *    - 当底盘被推离射击点时，以低速修正位置
 *
 * 底盘状态机：
 *   MOVING ──[到达射击点]──→ SHOOTING ──[敌人被击杀]──→ MOVING
 *   SHOOTING ──[被推远超过5s]──→ MOVING（重新导航）
 *   MOVING ──[超时]──→ 回退/跳过
 *
 * 四段变速策略（distance → speed）：
 *   distance > speed_switch_distance  → max_speed_far     全速冲刺
 *   distance > precise_mode_distance  → max_speed_near    减速接近
 *   distance > arrival_threshold      → 线性插值          平滑过渡
 *   distance ≤ arrival_threshold      → precise_adjust    精确微调
 *
 * 注意：target_locked 时会扩大变速距离（×2），让底盘更早减速，避免运动干扰瞄准。
 */

#include "movement_controller.h"

#include "data_manager.h"
#include "path_navigator.h"
#include "shared_context.h"
#include "shoot_controller.h"
#include "time_manager.h"
#include "waypoint.h"
#include "waypoint_utils.h"
using waypoint_utils::getEnemyIndex;
using waypoint_utils::kDistanceEpsilon;    // 距离 epsilon 0.001，防止除零
using waypoint_utils::kFastWaypointCount;  // 前4个路径点(P1-P4)使用快速到达阈值
using waypoint_utils::kNumEnemies;

MovementController::MovementController(SharedContext& ctx, const RuntimeConfig& config,
                                       rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr speed_pub,
                                       rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr angles_pub,
                                       DataManager& data_manager, PathNavigator& path_navigator,
                                       ShootController& shoot_controller, const std::vector<Waypoint>& waypoints,
                                       rclcpp::Logger logger)
    : ctx_(ctx),
      config_(config),
      speed_pub_(speed_pub),
      angles_pub_(angles_pub),
      data_manager_(data_manager),
      path_navigator_(path_navigator),
      shoot_controller_(shoot_controller),
      waypoints_(waypoints),
      logger_(logger) {}


/**
 * @brief 超时重试处理
 * @return true 表示已处理（调用方应直接return），false 表示未触发超时
 *
 * 逻辑：MOVING 状态下到达 move_timeout（默认10s）仍未到位：
 *   - 第1-2次：回退到上一个路径点重试
 *   - 第3次：放弃该路径点，跳到下一个
 */
bool MovementController::handleTimeoutRetry(size_t waypoint_idx, rclcpp::Time current_time) {
    if (!path_navigator_.isTimeout(current_time)) return false;
    {
        std::lock_guard<std::mutex> lock(ctx_.chassis.mutex);
        if (ctx_.chassis.value.state != SharedContext::MOVING) return false;
    }

    path_navigator_.incrementRetryCount(current_time);
    if (path_navigator_.isMaxRetriesExceeded()) {
        RCLCPP_INFO(logger_, "P%zu: 3次尝试失败，跳过该点位", waypoint_idx + 1);
        path_navigator_.skipCurrentPoint(current_time);
    } else {
        RCLCPP_INFO(logger_, "P%zu: 超时，返回上一位点（重试%d/3）", waypoint_idx + 1, path_navigator_.getRetryCount());
        path_navigator_.markReturnToPrevious();
    }
    return true;
}


/**
 * @brief SHOOTING 状态下漂移检测
 * @return true 表示已切换回 MOVING，false 表示无需处理
 *
 * 场景：射击期间底盘被敌方攻击推离射击点，距离超过 arrival_threshold × 2
 *       且持续超过5秒，则放弃射击，重新导航回到射击点
 *       （避免在远离目标的位置浪费弹药）
 */
bool MovementController::handleShootingDrift(size_t waypoint_idx, double distance, rclcpp::Time current_time) {
    {
        std::lock_guard<std::mutex> lock(ctx_.chassis.mutex);
        if (ctx_.chassis.value.state != SharedContext::SHOOTING) return false;
    }
    if (distance <= config_.arrival_threshold_normal * config_.drift_threshold_multiplier) return false;
    {
        std::lock_guard<std::mutex> lock(ctx_.gimbal.mutex);
        if (!TimeManager::isTimeout(ctx_.gimbal.value.shooting_start_time, current_time, config_.drift_timeout))
            return false;
    }

    RCLCPP_INFO(logger_, "P%zu: SHOOTING状态下被推得太远（%.2f），切换到MOVING状态重新导航", waypoint_idx + 1,
                distance);
    shoot_controller_.stopShooting();
    {
        std::lock_guard<std::mutex> lock(ctx_.chassis.mutex);
        ctx_.chassis.value.state = SharedContext::MOVING;
    }
    path_navigator_.resetTimeoutTimer(current_time);
    path_navigator_.resetRetryCount();
    return true;
}


/**
 * @brief 尝试从 MOVING 切换到 SHOOTING
 * @return true 表示被攻击打断（调用方应return），false 表示成功进入 SHOOTING
 *
 * 流程：
 *   1. 检查是否处于被攻击状态：如果是，且自身仍在掉血，则回退到攻击来源点位
 *   2. 停止底盘移动
 *   3. 设置云台角度为路径点预设值
 *   4. 获取目标敌人初始血量，启动射击控制器
 */
bool MovementController::tryStartShooting(size_t waypoint_idx, const Waypoint& target, rclcpp::Time current_time) {
    // 被攻击时的特殊处理：在 chassis_mutex 保护下判断和修改底盘状态
    // 先释放 chassis_mutex 再调用 moveToWaypoint（避免持锁调用外部模块）
    {
        std::lock_guard<std::mutex> lock(ctx_.chassis.mutex);
        if (ctx_.chassis.value.is_under_attack && !ctx_.chassis.value.attack_waypoint_stack.empty() &&
            waypoint_idx != ctx_.chassis.value.attack_waypoint_stack.top()) {
            int self_health_change = data_manager_.getSelfHealthChange();
            if (self_health_change < 0) {
                size_t retreat_idx = ctx_.chassis.value.attack_waypoint_stack.top();
                moveToWaypoint(retreat_idx, current_time);
                return true;
            }
            ctx_.chassis.value.is_under_attack = false;
            ctx_.chassis.value.attack_waypoint_stack.pop();
            RCLCPP_INFO(logger_, "P%zu: 自身血量不再减少，清除被攻击状态", waypoint_idx + 1);
        }
        ctx_.chassis.value.state = SharedContext::SHOOTING;
    }

    // 停止底盘移动，准备射击
    publishVelocity(0.0, 0.0, current_time);
    RCLCPP_INFO(logger_, "P%zu: 状态转换 MOVING -> SHOOTING", waypoint_idx + 1);

    // 设置云台初始角度（gimbal_mutex 保护）
    {
        std::lock_guard<std::mutex> gimbal_lock(ctx_.gimbal.mutex);
        ctx_.gimbal.value.shooting_start_time = current_time;
        ctx_.gimbal.value.current_yaw = target.yaw;
        ctx_.gimbal.value.current_pitch = target.pitch;
    }

    // 查找目标敌人在血量数组中的索引，获取初始血量
    int enemy_internal_index = getEnemyIndex(ctx_.player_id, waypoint_idx);
    int initial_enemy_health = 0;
    if (enemy_internal_index >= 0 && enemy_internal_index < kNumEnemies) {
        initial_enemy_health = data_manager_.getInitialEnemyHealth(enemy_internal_index);
    }
    shoot_controller_.startShooting(initial_enemy_health, current_time);

    ctx_.target_yaw = target.yaw;
    ctx_.target_pitch = target.pitch;
    RCLCPP_INFO(logger_, "到达P%zu，开始射击任务", waypoint_idx + 1);
    return false;
}


/**
 * @brief 跳过无射击任务的路径点，直接进入下一个
 *
 * P1-P3 没有射击任务（has_shoot_task=false），到达后直接跳过。
 * 跳转前先把云台角度预调到下一个路径点的 yaw/pitch，减少到位后的调整时间。
 */
void MovementController::advancePastWaypoint(size_t waypoint_idx, rclcpp::Time current_time) {
    size_t next_idx = waypoint_idx + 1;

    if (next_idx < waypoints_.size()) {
        ctx_.target_yaw = waypoints_[next_idx].yaw;
        ctx_.target_pitch = waypoints_[next_idx].pitch;
        double yaw, pitch;
        {
            std::lock_guard<std::mutex> gimbal_lock(ctx_.gimbal.mutex);
            ctx_.gimbal.value.current_yaw = ctx_.target_yaw;
            ctx_.gimbal.value.current_pitch = ctx_.target_pitch;
            yaw = ctx_.gimbal.value.current_yaw;
            pitch = ctx_.gimbal.value.current_pitch;
        }
        publishAngleCommand(yaw, pitch, false);

        RCLCPP_INFO(logger_, "离开P%zu，调整yaw/pitch到P%zu（%.1f, %.1f）", waypoint_idx + 1, next_idx + 1,
                    ctx_.target_yaw.load(), ctx_.target_pitch.load());
    }
    path_navigator_.moveToNext(current_time);
}


/**
 * @brief 主控制循环 —— 5ms 周期
 *
 * 核心决策树（按优先级）：
 *   1. 超时？ → 重试/跳过
 *   2. SHOOTING 被推远？ → 切回 MOVING
 *   3. MOVING 且到达 arrival_threshold 内？
 *      a. 未到 precise_arrival_threshold → 精确微调
 *      b. 已到位 + 有射击任务 → 进入 SHOOTING
 *      c. 已到位 + 无射击任务 → 跳到下一个路径点
 *   4. MOVING 且未到达 → 正常移动（四段变速）
 *   5. SHOOTING 且漂移 → 位置修正
 */
void MovementController::controlLoop() {
    auto current_time = TimeManager::getCurrentTime();

    size_t waypoint_idx = path_navigator_.getCurrentWaypointIndex();
    if (waypoint_idx >= waypoints_.size()) return;

    const Waypoint& target = waypoints_[waypoint_idx];
    auto [current_x, current_y] = data_manager_.getCurrentPosition();

    double dx = target.x - current_x;
    double dy = target.y - current_y;
    double distance = std::sqrt(dx * dx + dy * dy);

    // 优先级1：超时重试（内部有独立 chassis_mutex 保护）
    if (handleTimeoutRetry(waypoint_idx, current_time)) return;
    // 优先级2：射击漂移检测（内部有独立 chassis_mutex + gimbal_mutex 保护）
    if (handleShootingDrift(waypoint_idx, distance, current_time)) return;

    // 快照底盘状态（细粒度锁，避免持锁调用外部模块）
    SharedContext::State current_state;
    {
        std::lock_guard<std::mutex> lock(ctx_.chassis.mutex);
        current_state = ctx_.chassis.value.state;
    }

    // 到达阈值：前4个路径点(P1-P4)用快速阈值(2.0)，后续用普通阈值(0.8)
    double arrival_threshold =
        (waypoint_idx < kFastWaypointCount) ? config_.arrival_threshold_fast : config_.arrival_threshold_normal;

    // 精确到达阈值：快速路径点不区分精确/普通，直接用 fast 阈值
    double precise_arrival_threshold =
        (waypoint_idx < kFastWaypointCount) ? config_.arrival_threshold_fast : config_.precise_arrival_threshold;

    if (distance <= arrival_threshold && current_state == SharedContext::MOVING) {
        // 进入到达判定区域
        path_navigator_.resetTimeoutTimer(current_time);
        ctx_.is_precise_adjusting = false;

        if (distance > precise_arrival_threshold) {
            // 还没到精确阈值，执行精确微调（线性减速）
            ctx_.is_precise_adjusting = true;
            preciseAdjustment(dx, dy, distance);
            return;
        }

        // 完全到位，根据路径点类型决定下一步
        if (target.has_shoot_task) {
            if (tryStartShooting(waypoint_idx, target, current_time)) return;
        } else {
            advancePastWaypoint(waypoint_idx, current_time);
        }
    } else if (current_state == SharedContext::MOVING) {
        // 未到达，正常移动
        moveTowardsTarget(dx, dy, distance, target, waypoint_idx);
    } else if (current_state == SharedContext::SHOOTING && distance > config_.position_hold_threshold) {
        // SHOOTING 状态但偏离射击点，以低速修正位置
        double direction_x = dx / distance;
        double direction_y = dy / distance;
        double speed_ratio = std::min(1.0, distance / config_.arrival_threshold_normal);
        publishVelocity(direction_x * config_.precise_adjust_speed * speed_ratio,
                        direction_y * config_.precise_adjust_speed * speed_ratio, current_time);
    }
}


/**
 * @brief 四段变速移动 —— 从全速冲刺到精确微调的速度规划
 *
 * 速度曲线（distance → speed）：
 *
 *   distance > speed_switch_distance ──→ max_speed_far（全速）
 *       │
 *   distance > precise_mode_distance ──→ max_speed_near（减速）
 *       │
 *   distance > arrival_threshold ──→ 线性插值（平滑过渡）
 *       │
 *   distance ≤ arrival_threshold ──→ precise_adjust_speed × (dist/arrival)（精确微调）
 *
 * target_locked 时变速距离 ×2：让底盘更早开始减速，避免运动震动影响瞄准精度。
 * 快速路径点(P1-P4)的 speed_switch_distance 使用更大的 fast 值（10.0 vs 5.0），
 * 因为这些路径点只是路过，不需要太精细。
 */
void MovementController::moveTowardsTarget(double dx, double dy, double distance, const Waypoint& target,
                                           size_t waypoint_idx) {
    if (distance < kDistanceEpsilon) return;
    double direction_x = dx / distance;
    double direction_y = dy / distance;

    double speed_switch_distance = config_.speed_switch_distance;

    // 快速路径点使用更大的变速距离
    if (waypoint_idx < kFastWaypointCount) {
        speed_switch_distance = config_.speed_switch_distance_fast;
    }

    double precise_zone = config_.precise_mode_distance;
    if (ctx_.target_locked) {
        speed_switch_distance *= config_.lock_speed_multiplier;
        precise_zone *= config_.lock_speed_multiplier;
    }

    double current_max_speed;
    double arrival_threshold =
        (waypoint_idx < kFastWaypointCount) ? config_.arrival_threshold_fast : config_.arrival_threshold_normal;

    if (distance > speed_switch_distance) {
        current_max_speed = config_.max_speed_far;  // 远距离全速
    } else if (distance > precise_zone) {
        current_max_speed = config_.max_speed_near;  // 中距离减速
    } else if (distance > arrival_threshold) {
        // 过渡区：precise_adjust_speed → max_speed_near 线性插值
        double speed_ratio = (distance - arrival_threshold) / (precise_zone - arrival_threshold);
        current_max_speed =
            config_.precise_adjust_speed + (config_.max_speed_near - config_.precise_adjust_speed) * speed_ratio;
    } else {
        // 精确区：速度随距离线性衰减到 0
        double speed_ratio = distance / arrival_threshold;
        current_max_speed = config_.precise_adjust_speed * speed_ratio;
    }

    publishVelocity(direction_x * current_max_speed, direction_y * current_max_speed, TimeManager::getCurrentTime());
}


/**
 * @brief SHOOTING 状态下的位置保持循环
 *
 * 独立于 controlLoop 运行（由 position_adjust_timer 驱动，间隔可配置）。
 * 当底盘在射击期间被推离射击点（距离 > hold_threshold）时，
 * 以 precise_adjust_speed 低速修正位置。
 *
 * 注意：如果 controlLoop 正在执行精确微调(is_precise_adjusting=true)，
 * 本循环不执行，避免两个控制源同时发布速度指令产生冲突。
 */
void MovementController::positionAdjustLoop() {
    {
        std::lock_guard<std::mutex> lock(ctx_.chassis.mutex);
        if (ctx_.chassis.value.state != SharedContext::SHOOTING) return;
    }
    if (ctx_.is_precise_adjusting) return;  // 避免与 controlLoop 的精确微调冲突

    auto current_time = TimeManager::getCurrentTime();
    size_t waypoint_idx = path_navigator_.getCurrentWaypointIndex();
    if (waypoint_idx >= waypoints_.size()) return;

    const Waypoint& target = waypoints_[waypoint_idx];
    auto [current_x, current_y] = data_manager_.getCurrentPosition();

    double dx = target.x - current_x;
    double dy = target.y - current_y;
    double distance = std::sqrt(dx * dx + dy * dy);

    if (distance > config_.position_hold_threshold) {
        double direction_x = dx / distance;
        double direction_y = dy / distance;
        publishVelocity(direction_x * config_.precise_adjust_speed, direction_y * config_.precise_adjust_speed,
                        current_time);
    }
}


/**
 * @brief 精确微调 —— 到达 arrival_threshold 内但未到 precise_threshold 时的减速过渡
 * 速度 = precise_adjust_speed × (distance / arrival_threshold)
 * 随距离减小线性衰减，实现平滑停车
 */
void MovementController::preciseAdjustment(double dx, double dy, double distance) {
    double direction_x = dx / distance;
    double direction_y = dy / distance;
    double speed_ratio = distance / config_.arrival_threshold_normal;
    double adjust_speed = config_.precise_adjust_speed * speed_ratio;

    publishVelocity(direction_x * adjust_speed, direction_y * adjust_speed, TimeManager::getCurrentTime());
}

void MovementController::moveToWaypoint(size_t waypoint_idx, rclcpp::Time current_time) {
    RCLCPP_INFO(logger_, "开始导航到 P%zu", waypoint_idx + 1);
    path_navigator_.navigateToWaypoint(waypoint_idx, current_time);
}

/**
 * @brief 发布底盘速度指令（TwistStamped 消息）
 * @param vx, vy 世界坐标系下的速度分量（地图单位/s）
 */
void MovementController::publishVelocity(double vx, double vy, rclcpp::Time stamp) {
    auto message = geometry_msgs::msg::TwistStamped();
    message.header.stamp = stamp;
    message.header.frame_id = "base_link";
    message.twist.linear.x = vx;
    message.twist.linear.y = vy;
    speed_pub_->publish(message);
}

/**
 * @brief 发布云台角度指令（SendData 消息）
 * 用于在路径点切换时预调云台朝向，不触发射击(if_shoot=false)
 */
void MovementController::publishAngleCommand(double yaw, double pitch, bool if_shoot) {
    auto message = tdt_interface::msg::SendData();
    message.yaw = yaw;
    message.pitch = pitch;
    message.if_shoot = if_shoot;
    angles_pub_->publish(message);
}
