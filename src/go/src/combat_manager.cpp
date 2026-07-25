/**
 * @file combat_manager.cpp
 * @brief 战斗管理器 —— 自瞄检测、目标跟踪、弹道补偿的核心循环
 *
 * 本模块运行两个独立定时循环：
 *
 * 1. shootLoop()（每30ms）：
 *    检测 → 距离筛选 → 跟踪状态机 → EKF滤波/预测 → 弹道补偿 → 更新云台角度
 *
 * 2. swingLoop()（每20ms）：
 *    根据 ShootController 的模式决定云台朝向：
 *    - SWING 模式：在硬编码 yaw 基础上叠加三角波摆动
 *    - HARDCODE 模式：直接指向路径点预设角度
 *    - AUTO 模式：不干预（由 shootLoop 控制）
 *    最终统一发布射击指令
 *
 * 距离门控机制（Distance Gate）：
 *   防止 YOLO 检测的距离跳变导致云台抖动。
 *   - 首次检测通过后记录 last_accepted_distance
 *   - 后续检测的距离与上次偏差 > config_.distance_gate_acceptance 时丢弃
 *   - 切换到 LOST 状态后重置门控，允许重新接受新距离
 *
 * 降级策略：
 *   连续检测失败 ≥ detection_fail_max 次后，放弃自瞄，云台切回路径点预设角度，
 *   并重置 EKF 状态。这是为了应对模型持续误检或目标确实消失的场景。
 */

#include "combat_manager.h"

#include <limits>

#include "path_navigator.h"
#include "performance_monitor.h"
#include "shared_context.h"
#include "shoot_controller.h"
#include "time_manager.h"
#include "waypoint.h"
#include "waypoint_utils.h"
#include "yolov5_detector.h"

CombatManager::CombatManager(SharedContext& ctx, const RuntimeConfig& config,
                             rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr angles_pub,
                             YOLOv5Detector& detector, PerformanceMonitor& perf_monitor, PathNavigator& path_navigator,
                             ShootController& shoot_controller, const std::vector<Waypoint>& waypoints,
                             rclcpp::Logger logger)
    : ctx_(ctx),
      config_(config),
      angles_pub_(angles_pub),
      detector_(detector),
      perf_monitor_(perf_monitor),
      path_navigator_(path_navigator),
      shoot_controller_(shoot_controller),
      waypoints_(waypoints),
      logger_(logger),
      target_ekf_({config_.ekf_q_yaw, config_.ekf_q_pitch, config_.ekf_q_distance, config_.ekf_r_yaw,
                   config_.ekf_r_pitch, config_.ekf_r_distance_denom, config_.ekf_r_distance_base, config_.ekf_p0_yaw,
                   config_.ekf_p0_pitch, config_.ekf_p0_distance, config_.ekf_p_reset, config_.ekf_q_adaptive_alpha,
                   config_.ekf_q_adaptive_max_scale, config_.nis_chi2_upper, config_.nis_window}),
      tracking_sm_(config_.tracking_min_detect, config_.tracking_max_temp_lost, config_.tracking_nis_failure_max) {}


/**
 * @brief 主射击循环 —— 检测、跟踪、滤波、补偿的完整流水线
 *
 * 执行频率：约30ms（由 GoNode 的 shoot_timer 驱动）
 *
 * 流程概要：
 *   1. 计算 dt（两次调用的时间间隔，用于 EKF 预测）
 *   2. YOLO 推理检测所有装甲板
 *   3. 距离门控筛选，选最近的目标
 *   4. 跟踪状态机判断当前跟踪状态
 *   5. 根据状态执行不同策略：
 *      - LOST：放弃目标，累计失败，必要时降级
 *      - TEMP_LOST：用 EKF 预测维持瞄准
 *      - DETECTING/TRACKING：EKF 更新，弹道补偿，锁定目标
 */
void CombatManager::shootLoop() {
    auto current_time = TimeManager::getCurrentTime();

    size_t waypoint_idx = path_navigator_.getCurrentWaypointIndex();
    if (waypoint_idx >= waypoints_.size()) return;

    const Waypoint& target = waypoints_[waypoint_idx];

    // ── Step 1: 计算 dt ──
    double dt = config_.combat_ekf_default_dt;
    auto now = std::chrono::steady_clock::now();
    if (last_ekf_time_set_) {
        dt = std::chrono::duration<double>(now - last_ekf_time_).count();
        if (dt > config_.combat_ekf_max_dt) dt = config_.combat_ekf_default_dt;
    }
    last_ekf_time_ = now;
    last_ekf_time_set_ = true;

    // ── Step 2: YOLO 推理检测 ──
    perf_monitor_.startTimer("inference");
    auto armors = detector_.detectArmors();

    // ── Step 3: 距离门控筛选 ──
    // 对所有检测到的装甲板进行距离过滤，选最近且通过门控的作为跟踪目标
    double distance = config_.default_detection_distance;
    if (!armors.empty()) {
        size_t closest_idx = 0;
        double min_raw_distance = std::numeric_limits<double>::max();
        bool any_accepted = false;
        for (size_t i = 0; i < armors.size(); ++i) {
            double raw = detector_.rawPnPDistance(armors[i].armorBbox);
            if (raw > config_.distance_gate_max) continue;  // 距离过远，直接丢弃
            // 距离门控：偏差超过门控接受范围的检测认为是跳变，丢弃
            if (ctx_.apply_distance_gate &&
                std::abs(raw - ctx_.last_accepted_distance) > config_.distance_gate_acceptance)
                continue;
            any_accepted = true;
            if (raw < min_raw_distance) {
                min_raw_distance = raw;
                closest_idx = i;
            }
        }
        if (!any_accepted) {
            armors.clear();  // 所有检测都被门控过滤
        } else {
            // 将最近的目标换到 armors[0] 位置
            if (closest_idx != 0) std::swap(armors[0], armors[closest_idx]);
            // 用 PnP 精确测距（带中值滤波平滑）
            std::tie(distance, std::ignore) = detector_.solvePnPDistance(armors[0].armorBbox);
            armors[0].distance = static_cast<float>(distance);
            ctx_.last_accepted_distance = distance;
            ctx_.apply_distance_gate = true;
        }
    }
    perf_monitor_.endTimer("inference");

    detector_.cacheDetections(armors.empty() ? std::vector<DetectedArmor>{} : std::vector<DetectedArmor>{armors[0]});

    if (config_.combat_debug_log_interval > 0 && ++detect_debug_count_ % config_.combat_debug_log_interval == 0) {
        RCLCPP_DEBUG(logger_, "检测到 %zu 个目标", armors.size());
    }

    // ── Step 4: 跟踪状态机 ──
    // 输入：是否检测到目标 + EKF的NIS失败率
    // 状态流转：LOST → DETECTING(连续3次) → TRACKING → TEMP_LOST(最多15帧) → LOST
    TrackingState prev_state = tracking_sm_.state();
    tracking_sm_.update(!armors.empty(), target_ekf_.nisFailureRate());
    TrackingState curr_state = tracking_sm_.state();
    if (prev_state != curr_state) {
        static const char* state_names[] = {"LOST", "DETECTING", "TRACKING", "TEMP_LOST"};
        RCLCPP_DEBUG(logger_, "跟踪状态: %s → %s", state_names[static_cast<int>(prev_state)],
                     state_names[static_cast<int>(curr_state)]);
    }

    // ── Step 5: 根据跟踪状态执行不同策略 ──

    if (curr_state == TrackingState::LOST) {
        // 完全丢失目标：解除锁定，累计失败计数
        ctx_.target_locked.store(false);
        if (armors.empty()) {
            ctx_.detection_fail_count++;
            // 连续失败超过阈值 → 降级到硬编码角度，放弃自瞄
            if (ctx_.detection_fail_count >= config_.detection_fail_max) {
                if (!degradation_warned_) {
                    RCLCPP_WARN(logger_, "连续检测失败%d次，启用降级策略使用硬编码角度",
                                ctx_.detection_fail_count.load());
                    degradation_warned_ = true;
                }
                {
                    std::lock_guard<std::mutex> lock(ctx_.gimbal.mutex);
                    ctx_.gimbal.value.current_yaw = target.yaw;
                    ctx_.gimbal.value.current_pitch = target.pitch;
                }
                target_ekf_.reset();  // 重置滤波器，为下次重新锁定做准备
                ctx_.detection_fail_count = 0;
            }
        }

    } else if (curr_state == TrackingState::TEMP_LOST) {
        // 短暂丢失（目标被遮挡或检测闪烁）：用 EKF 预测维持瞄准
        auto [yaw, pitch, dist] = target_ekf_.predict(dt);

        ctx_.target_locked.store(true);
        {
            std::lock_guard<std::mutex> lock(ctx_.gimbal.mutex);
            // EKF 输出单位为弧度，转换为角度后计算弹道补偿
            double yaw_deg = yaw * 180.0 / CV_PI;
            double pitch_deg = pitch * 180.0 / CV_PI;
            auto [final_yaw, final_pitch] = detector_.calculateBallisticCompensation(
                yaw_deg, pitch_deg, dist, ctx_.gimbal.value.current_yaw, ctx_.gimbal.value.current_pitch);
            ctx_.gimbal.value.current_yaw = final_yaw;
            ctx_.gimbal.value.current_pitch = final_pitch;
        }

    } else {
        // DETECTING 或 TRACKING 状态：正常自瞄流程
        ctx_.detection_fail_count = 0;
        degradation_warned_ = false;

        // 从 LOST 切回时，重置距离门控以接受新的距离基准
        if (prev_state == TrackingState::LOST) {
            ctx_.apply_distance_gate = false;
        }

        // PnP 求解 yaw/pitch/distance（弧度），送入 EKF 滤波
        auto [raw_yaw, raw_pitch, raw_distance] = detector_.solvePnPYawPitchDist(armors[0].armorBbox);
        auto [yaw, pitch, dist] = target_ekf_.update(dt, raw_yaw, raw_pitch, raw_distance);

        ctx_.target_locked.store(true);
        {
            std::lock_guard<std::mutex> lock(ctx_.gimbal.mutex);
            ctx_.gimbal.value.last_detection_time = current_time;
            double yaw_deg = yaw * 180.0 / CV_PI;
            double pitch_deg = pitch * 180.0 / CV_PI;
            auto [final_yaw, final_pitch] = detector_.calculateBallisticCompensation(
                yaw_deg, pitch_deg, dist, ctx_.gimbal.value.current_yaw, ctx_.gimbal.value.current_pitch);
            ctx_.gimbal.value.current_yaw = final_yaw;
            ctx_.gimbal.value.current_pitch = final_pitch;
        }
    }
}


/**
 * @brief 摆动/射击指令循环 —— 决定云台朝向并发布射击命令
 *
 * 执行频率：约20ms（由 GoNode 的 swing_timer 驱动）
 *
 * 与 shootLoop() 的分工：
 *   - shootLoop 负责"看到目标"：检测、滤波、更新 gimbal.current_yaw/pitch
 *   - swingLoop 负责"发布指令"：根据射击模式决定最终 yaw/pitch，组装并发送 SendData
 *
 * 射击模式对云台朝向的影响：
 *   - SWING 模式：在路径点预设 yaw 上叠加摆动偏移（三角波扫描）
 *   - 非AUTO模式（HARDCODE）：直接使用路径点预设的 yaw/pitch
 *   - AUTO 模式：不修改 gimbal 角度（由 shootLoop 的 EKF 输出控制）
 */
void CombatManager::swingLoop() {
    // 计算 dt，用于摆动角度增量计算
    auto now = std::chrono::steady_clock::now();
    if (!ctx_.last_swing_time_set) {
        ctx_.last_swing_time = now;
        ctx_.last_swing_time_set = true;
    }
    double dt = std::chrono::duration<double>(now - ctx_.last_swing_time).count();
    ctx_.last_swing_time = now;
    auto current_time = TimeManager::getCurrentTime();

    size_t waypoint_idx = path_navigator_.getCurrentWaypointIndex();
    if (waypoint_idx >= waypoints_.size()) return;

    const Waypoint& target = waypoints_[waypoint_idx];

    // 根据射击模式设置云台朝向
    if (shoot_controller_.isSwinging()) {
        // SWING：在路径点 yaw 基础上叠加三角波摆动
        double swing_yaw = shoot_controller_.updateSwingYaw(dt);
        std::lock_guard<std::mutex> lock(ctx_.gimbal.mutex);
        ctx_.gimbal.value.current_yaw = target.yaw + swing_yaw;
        ctx_.gimbal.value.current_pitch = target.pitch;
    } else if (!shoot_controller_.isAutoAimEnabled()) {
        // HARDCODE 模式：直接指向路径点预设角度
        // 注意：AUTO 模式下不进入此分支，保留 shootLoop 写入的 EKF 角度
        std::lock_guard<std::mutex> lock(ctx_.gimbal.mutex);
        ctx_.gimbal.value.current_yaw = target.yaw;
        ctx_.gimbal.value.current_pitch = target.pitch;
    }

    RCLCPP_DEBUG(logger_, "P%zu: 发布射击命令", waypoint_idx + 1);
    publishShootCommand(waypoint_idx, current_time);
}


/**
 * @brief 组装并发布 SendData 射击指令
 *
 * 角度选取逻辑（优先级从高到低）：
 *   1. SWING/HARDCODE 模式：使用 gimbal 当前角度（已由 swingLoop 设置）
 *   2. AUTO 模式：使用 gimbal 当前角度（已由 shootLoop 的 EKF 输出设置）
 *   3. 都不是：回退到路径点预设角度（理论上不会走到这里）
 *
 * if_shoot 判定条件（必须同时满足）：
 *   - target_locked：当前有被锁定的目标（检测/跟踪/临时丢失均算锁定）
 *   - 检测未过时：last_detection_time 距今不超过 detection_fail_max × 30ms
 *     （防止目标已消失很久但仍因 target_locked 残留而持续发弹）
 */
void CombatManager::publishShootCommand(size_t waypoint_idx, rclcpp::Time current_time) {
    auto message = tdt_interface::msg::SendData();
    double yaw, pitch;
    bool if_shoot;

    {
        std::lock_guard<std::mutex> lock(ctx_.gimbal.mutex);
        if (shoot_controller_.isSwinging() || shoot_controller_.isAutoAimEnabled()) {
            yaw = ctx_.gimbal.value.current_yaw;
            pitch = ctx_.gimbal.value.current_pitch;
        } else {
            // PRESET 回退：使用 YAML 路径点中的预设云台角度
            yaw = waypoints_[waypoint_idx].yaw;
            pitch = waypoints_[waypoint_idx].pitch;
        }

        // 检测时效性判断：超过阈值时间没有新检测结果则停止发弹
        auto detection_age = current_time - ctx_.gimbal.value.last_detection_time;
        bool detection_stale = detection_age.seconds() > config_.detection_fail_max * 0.030;
        if_shoot = ctx_.target_locked && !detection_stale;
    }

    message.yaw = yaw;
    message.pitch = pitch;
    message.if_shoot = if_shoot;
    angles_pub_->publish(message);
}
