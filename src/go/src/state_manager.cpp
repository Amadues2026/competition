/**
 * @file state_manager.cpp
 * @brief 状态管理器 —— 血量监控、自瞄模式决策、被攻击响应
 *
 * 本模块是射击决策的"大脑"，运行 healthCheckLoop()（1秒周期），负责：
 *
 * 1. 血量变化评估 → 判断射击是否有效命中
 * 2. 自瞄模式降级链：AUTO → HARDCODE → SWING
 * 3. 敌人击杀处理 → SHOOTING → MOVING
 * 4. 被攻击响应 → 跳转到下一射击点 / 优先完成即将击杀的目标
 *
 * 与其他模块的关系：
 *   - ShootController：提供射击模式状态，接受模式切换指令
 *   - DataManager：获取敌我血量变化数据
 *   - PathNavigator：控制路径跳转
 *   - MovementController：通过 SharedContext 间接影响底盘状态
 *
 * 被攻击响应策略（两种场景）：
 *   - 基地区域(P7-P11)被攻击：立即跳到基地区域的下一个循环点位（保持机动）
 *   - 非基地区域(P4-P6)被攻击：
 *     a. 目标血量 ≤ 20% → 继续打，不撤（快死了不放弃）
 *     b. 目标血量 > 20% → 跳到下一个射击任务点，原点位压栈（稍后可回来）
 */

#include "state_manager.h"

#include "data_manager.h"
#include "path_navigator.h"
#include "performance_monitor.h"
#include "shared_context.h"
#include "shoot_controller.h"
#include "time_manager.h"
#include "waypoint.h"
#include "waypoint_utils.h"

using waypoint_utils::kBaseLoopStart;            // 6（基地区域第一个路径点P7的索引）
using waypoint_utils::kBaseWaypointEnd;          // 10（基地区域最后一个路径点P11的索引）
using waypoint_utils::kFirstShootWaypointIndex;  // 4（第一个射击路径点P5的索引）
using waypoint_utils::kNumEnemies;               // 6个敌人单位
// kHealthRatioEscape 已移至 YAML（config_.waypoint_health_ratio_escape）
using waypoint_utils::findNextShootTaskIndex;
using waypoint_utils::getEnemyIndex;
using waypoint_utils::getHealthChangeThreshold;

StateManager::StateManager(SharedContext& ctx, ShootController& shoot_controller, DataManager& data_manager,
                           PathNavigator& path_navigator, PerformanceMonitor* perf_monitor,
                           const waypoint_utils::HealthChangeThresholdConfig& health_thresholds,
                           const RuntimeConfig& config, const std::vector<Waypoint>& waypoints,
                           rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr angles_pub, rclcpp::Logger logger)
    : ctx_(ctx),
      shoot_controller_(shoot_controller),
      data_manager_(data_manager),
      path_navigator_(path_navigator),
      perf_monitor_(perf_monitor),
      health_thresholds_(health_thresholds),
      config_(config),
      waypoints_(waypoints),
      angles_pub_(angles_pub),
      logger_(logger) {}


/**
 * @brief 周期性打印性能报告
 * 每10次 healthCheckLoop（即10秒）输出一次推理耗时等性能指标
 */
void StateManager::tryReportPerformance() {
    health_check_count_++;
    if (health_check_count_ >= 10) {
        health_check_count_ = 0;
        if (perf_monitor_) {
            perf_monitor_->reportAll();
        }
    }
}


/**
 * @brief 核心决策函数 —— 血量评估 + 自瞄模式切换 + 被攻击/击杀检测
 * @param waypoint_idx 当前射击路径点索引
 * @param current_time 当前ROS2时间
 *
 * 决策流程（按顺序执行，不互斥）：
 *
 * 1. 获取敌方血量数据，计算血量变化
 * 2. checkHealthChange：评估是否命中，更新 miss 计数
 * 3. shouldDisableAutoAim：AUTO 连续 miss ≥ 10 → 切 HARDCODE
 * 4. shouldStartSwinging：HARDCODE 连续 miss ≥ 3 → 切 SWING
 * 5. SWING 期间命中：停止摆动，恢复历史 yaw（回到上次有效位置）
 * 6. 自身被攻击：根据场景决定是否撤退
 * 7. 敌人被击杀：清理状态，前进到下一点位
 */
void StateManager::evaluateHealthAndAutoAim(size_t waypoint_idx, rclcpp::Time current_time) {
    // ── Step 1: 获取血量数据 ──
    int health_change_threshold = getHealthChangeThreshold(health_thresholds_, waypoint_idx);
    int enemy_internal_index = getEnemyIndex(ctx_.player_id, waypoint_idx);

    int current_enemy_health = 0;
    int initial_enemy_health = 0;
    int health_change = 0;
    bool enemy_health_valid = false;

    if (enemy_internal_index >= 0 && enemy_internal_index < kNumEnemies) {
        current_enemy_health = data_manager_.getCurrentEnemyHealth(enemy_internal_index);
        initial_enemy_health = data_manager_.getInitialEnemyHealth(enemy_internal_index);
        health_change = data_manager_.getEnemyHealthChange(enemy_internal_index);
        enemy_health_valid = data_manager_.isEnemyHealthInitialized(enemy_internal_index);
    }

    RCLCPP_INFO(logger_, "P%zu: 血量变化=%d, 阈值=%d, 连续未达标=%d", waypoint_idx + 1, health_change,
                health_change_threshold, shoot_controller_.getConsecutiveMissCountForSwing());

    // ── Step 2: 评估命中 ──
    // health_change < threshold → 命中有效，重置 miss 计数
    // health_change ≥ threshold → 未命中，累加 miss 计数
    bool health_changed = shoot_controller_.checkHealthChange(health_change, health_change_threshold, current_time);
    if (health_changed) {
        RCLCPP_INFO(logger_, "P%zu: 血量变化达到明显标准，重置计数", waypoint_idx + 1);
    }

    // ── Step 3: 自瞄禁用判定（AUTO → HARDCODE）──
    // 连续 miss 达到 disable 阈值（默认10次），自瞄无法打中目标
    if (shoot_controller_.shouldDisableAutoAim(current_time)) {
        shoot_controller_.disableAutoAim();
        RCLCPP_INFO(logger_, "P%zu: 禁用自瞄，切换到硬编码", waypoint_idx + 1);
    }

    // ── Step 4: 摆动触发判定（HARDCODE → SWING）──
    // 硬编码角度也打不中时，开始左右摆动扫描
    if (shoot_controller_.shouldStartSwinging(current_time)) {
        shoot_controller_.startSwinging(current_time);
        RCLCPP_INFO(logger_, "P%zu: 连续未达标触发摆动", waypoint_idx + 1);
    }

    // ── Step 5: 摆动期间命中处理（SWING → HARDCODE）──
    // 摆动过程中如果敌方血量开始变化，说明扫到了目标
    // 停止摆动，把 yaw 恢复到历史有效值（1秒前的角度，大概率是目标方向）
    if (shoot_controller_.isSwinging() && health_changed) {
        shoot_controller_.stopSwinging(current_time);
        double prev_yaw;
        std::tie(prev_yaw, std::ignore) = data_manager_.getHistoricalYawPitch(current_time);
        {
            std::lock_guard<std::mutex> lock(ctx_.gimbal.mutex);
            ctx_.gimbal.value.current_yaw = prev_yaw;
        }
        RCLCPP_INFO(logger_, "P%zu: 检测到血量变化，停止摆动，调整yaw到前一秒数值", waypoint_idx + 1);
    }

    // ── Step 6: 自身被攻击检测 ──
    // self_health_change < 0 表示自身在掉血（被敌方攻击）
    int self_health_change = data_manager_.getSelfHealthChange();
    if (self_health_change < 0 && enemy_health_valid && initial_enemy_health > 0) {
        RCLCPP_INFO(logger_, "P%zu: 检测到自身血量减少%d，被攻击", waypoint_idx + 1, self_health_change);
        handleSelfUnderAttack(waypoint_idx, current_enemy_health, initial_enemy_health);
    }

    // ── Step 7: 敌人击杀检测 ──
    // 敌方血量归零 → 目标被摧毁，前进到下一个路径点
    if (enemy_health_valid && current_enemy_health <= 0) {
        handleKilledEnemy(waypoint_idx, current_time);
    }
}


/**
 * @brief 敌人被击杀后的处理
 *
 * 清理工作：
 *   - 停止射击
 *   - 底盘切回 MOVING 状态
 *   - 清除被攻击状态和攻击回退栈
 *   - 取消路径回退（如果正在回退）
 *   - 把云台 yaw/pitch 预调到下一个路径点
 *   - 导航到下一个路径点
 */
void StateManager::handleKilledEnemy(size_t waypoint_idx, rclcpp::Time current_time) {
    RCLCPP_INFO(logger_, "P%zu: 敌人击杀，状态转换 SHOOTING -> MOVING", waypoint_idx + 1);
    shoot_controller_.stopShooting();
    {
        std::lock_guard<std::mutex> lock(ctx_.chassis.mutex);
        ctx_.chassis.value.state = SharedContext::MOVING;
        ctx_.chassis.value.is_under_attack = false;
        while (!ctx_.chassis.value.attack_waypoint_stack.empty()) {
            ctx_.chassis.value.attack_waypoint_stack.pop();  // 清空回退栈
        }
    }
    if (path_navigator_.isReturning()) {
        path_navigator_.cancelReturn();  // 取消正在进行的回退
    }

    // 预调云台到下一个路径点的朝向
    size_t next_idx = waypoint_idx + 1;
    if (next_idx < waypoints_.size()) {
        ctx_.target_yaw = waypoints_[next_idx].yaw;
        ctx_.target_pitch = waypoints_[next_idx].pitch;
        double yaw, pitch;
        {
            std::lock_guard<std::mutex> lock(ctx_.gimbal.mutex);
            yaw = ctx_.gimbal.value.current_yaw = ctx_.target_yaw;
            pitch = ctx_.gimbal.value.current_pitch = ctx_.target_pitch;
        }

        publishAngleCommand(yaw, pitch, false);

        RCLCPP_INFO(logger_, "P%zu: 目标被摧毁，调整yaw/pitch到P%zu（%.1f, %.1f）", waypoint_idx + 1, next_idx + 1,
                    ctx_.target_yaw.load(), ctx_.target_pitch.load());
    }

    path_navigator_.moveToNext(current_time);
    RCLCPP_INFO(logger_, "P%zu: 目标被摧毁，前往下一点", waypoint_idx + 1);
}


/**
 * @brief 血量检查主循环 —— 1秒周期
 *
 * 前置条件：
 *   - 必须正在射击（isShooting() == true）
 *   - 当前路径点必须是射击任务点（has_shoot_task == true）
 * 不满足条件时直接返回，不做任何判断
 *
 * 每10次调用还会触发一次性能报告打印
 */
void StateManager::healthCheckLoop() {
    tryReportPerformance();

    if (!shoot_controller_.isShooting()) return;

    auto current_time = TimeManager::getCurrentTime();
    size_t waypoint_idx = path_navigator_.getCurrentWaypointIndex();

    // 非射击路径点（P1-P3）不做血量评估
    if (waypoint_idx >= waypoints_.size() || !waypoints_[waypoint_idx].has_shoot_task) {
        return;
    }

    evaluateHealthAndAutoAim(waypoint_idx, current_time);
}


/**
 * @brief 被攻击响应 —— 两种场景两种策略
 * @param waypoint_idx 当前路径点
 * @param current_enemy_health 目标当前血量
 * @param initial_enemy_health 目标初始血量
 *
 * 场景A：基地区域(P7-P11)被攻击
 *   → 直接跳到基地区域的下一个循环点位（P7→P8→...→P11→P7）
 *   → 原因：基地区域是固定路线绕圈，被打说明位置暴露，立即换位
 *
 * 场景B：非基地区域(P4-P6)被攻击
 *   a. 目标血量 ≤ 20% → 不撤退（kHealthRatioEscape=0.2）
 *      原因：目标快死了，优先击杀比撤退更有价值
 *   b. 目标血量 > 20% → 跳到下一个射击任务点
 *      原因：目标血量充足，说明短时间打不死，先撤保存血量
 *      当前点位压入 attack_waypoint_stack，后续可以回来继续打
 *
 * attack_waypoint_stack 的作用：
 *   记录"被打断的射击任务"，等自身不再掉血后，
 *   MovementController::tryStartShooting 会检查栈顶，
 *   如果到达的目标点和栈顶不同且自身仍在掉血，则回退到栈顶点位
 */
void StateManager::handleSelfUnderAttack(size_t waypoint_idx, int current_enemy_health, int initial_enemy_health) {
    auto current_time = TimeManager::getCurrentTime();

    // ── 场景A：基地区域被攻击 → 跳到下一个循环点位 ──
    if (waypoint_idx >= kBaseLoopStart && waypoint_idx <= kBaseWaypointEnd) {
        size_t next_shoot_idx = waypoint_idx + 1;
        if (next_shoot_idx > kBaseWaypointEnd) {
            next_shoot_idx = kBaseLoopStart;  // P11之后回到P7
        }

        RCLCPP_INFO(logger_, "P%zu: 被攻击，立即前往下一点位P%zu", waypoint_idx + 1, next_shoot_idx + 1);

        shoot_controller_.stopShooting();
        {
            std::lock_guard<std::mutex> lock(ctx_.chassis.mutex);
            ctx_.chassis.value.state = SharedContext::MOVING;
            ctx_.chassis.value.is_under_attack = false;
        }

        moveToWaypoint(next_shoot_idx, current_time);
        return;
    }

    // ── 场景B：非基地区域被攻击 ──
    if (initial_enemy_health <= 0) {
        return;  // 无法计算血量比，跳过
    }

    // 目标血量百分比：当前/初始
    double health_ratio = (double)current_enemy_health / initial_enemy_health;

    // 目标血量不足20% → 优先击杀，不撤退
    if (health_ratio <= config_.waypoint_health_ratio_escape) {
        RCLCPP_INFO(logger_, "P%zu: 被攻击，但目标血量不足20%%(%.1f%%)，继续打击任务", waypoint_idx + 1,
                    health_ratio * 100);
        return;
    }

    // 目标血量充足 → 撤退，跳到下一个射击任务点
    size_t next_shoot_idx = findNextShootTaskIndex(waypoints_, waypoint_idx);

    if (next_shoot_idx == waypoint_idx) {
        next_shoot_idx = kFirstShootWaypointIndex;  // 防止找不到下一个射击点
    }

    RCLCPP_INFO(logger_, "P%zu: 被攻击，目标血量%.1f%%，前往下一射击任务点P%zu", waypoint_idx + 1, health_ratio * 100,
                next_shoot_idx + 1);

    // 压栈：记录被打断的位置，以便后续可能返回
    {
        std::lock_guard<std::mutex> lock(ctx_.chassis.mutex);
        if (ctx_.chassis.value.attack_waypoint_stack.empty() ||
            ctx_.chassis.value.attack_waypoint_stack.top() != waypoint_idx) {
            ctx_.chassis.value.attack_waypoint_stack.push(waypoint_idx);
        }
        ctx_.chassis.value.is_under_attack = true;
    }

    moveToWaypoint(next_shoot_idx, current_time);
}


void StateManager::publishAngleCommand(double yaw, double pitch, bool if_shoot) {
    auto message = tdt_interface::msg::SendData();
    message.yaw = yaw;
    message.pitch = pitch;
    message.if_shoot = if_shoot;
    angles_pub_->publish(message);
}

void StateManager::moveToWaypoint(size_t waypoint_idx, rclcpp::Time current_time) {
    RCLCPP_INFO(logger_, "开始导航到 P%zu", waypoint_idx + 1);
    path_navigator_.navigateToWaypoint(waypoint_idx, current_time);
}
