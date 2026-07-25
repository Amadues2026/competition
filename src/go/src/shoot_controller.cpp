/**
 * @file shoot_controller.cpp
 * @brief 射击模式控制器
 *
 * 管理三种射击模式之间的切换：
 *   AUTO（自瞄）→ HARDCODE（硬编码角度）→ SWING（摆动扫描）
 *
 * 核心逻辑：通过血量变化检测判断射击是否有效命中，连续未命中则逐步降级：
 *   1. AUTO 模式下连续 miss ≥ disable阈值(10) → 降级为 HARDCODE
 *   2. HARDCODE 模式下连续 miss ≥ swing阈值(3) → 降级为 SWING（左右摆动扫描目标区域）
 *   3. SWING 期间一旦检测到血量变化（命中）→ 回退到 HARDCODE 并重新尝试
 *
 * 线程安全：所有公开方法均通过 mutex_ 保护，可被 healthCheckLoop 和 swingLoop 并发调用。
 */

#include "shoot_controller.h"

/** @brief 构造函数，接收由组装层完成加载和验证的参数。 */
ShootController::ShootController(const ShootControllerConfig& config)
    : aim_mode_(AimMode::HARDCODE),
      is_shooting_(false),
      consecutive_miss_count_for_disable_(0),
      consecutive_miss_count_for_swing_(0),
      swing_offset_yaw_(0),
      swing_direction_(1),
      logger_(rclcpp::get_logger("ShootController")),
      swing_range_(config.swing_range),
      swing_speed_(config.swing_speed),
      consecutive_miss_threshold_disable_(config.consecutive_miss_disable),
      consecutive_miss_threshold_swing_(config.consecutive_miss_swing) {}

/**
 * @brief 开始射击任务（到达射击路径点时调用）
 * @param initial_health 目标敌人的初始血量（记录用于后续血量变化计算）
 * @param current_time 当前ROS2时间
 *
 * 效果：启用自瞄模式(AUTO)，重置连续miss计数器
 */
void ShootController::startShooting(int initial_health, rclcpp::Time current_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    is_shooting_ = true;
    aim_mode_ = AimMode::AUTO;
    consecutive_miss_count_for_disable_ = 0;
    consecutive_miss_count_for_swing_ = 0;
    RCLCPP_INFO(logger_, "启用自瞄，初始血量: %d", initial_health);
}

/**
 * @brief 停止射击（敌人被击杀、切换路径点、被攻击撤退时调用）
 * 重置所有状态，回到 HARDCODE 模式
 */
void ShootController::stopShooting() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_shooting_ = false;
    aim_mode_ = AimMode::HARDCODE;
    consecutive_miss_count_for_disable_ = 0;
    consecutive_miss_count_for_swing_ = 0;
    swing_offset_yaw_ = 0.0;
}

/**
 * @brief 手动禁用自瞄，切换到硬编码角度模式
 * 在 AUTO 模式下连续 miss 达到阈值时由 StateManager 调用
 * 注意：只重置 swing 的 miss 计数（因为接下来要进入 swing 判断逻辑）
 */
void ShootController::disableAutoAim() {
    std::lock_guard<std::mutex> lock(mutex_);
    aim_mode_ = AimMode::HARDCODE;
    consecutive_miss_count_for_swing_ = 0;
}

bool ShootController::isShooting() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_shooting_;
}
bool ShootController::isAutoAimEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return aim_mode_ == AimMode::AUTO;
}

/**
 * @brief 评估本次射击是否有效命中
 * @param health_change 目标敌人的血量变化值（5秒前血量 - 当前血量，正值表示敌方掉血）
 * @param threshold 该路径点的血量变化阈值（不同目标阈值不同）
 * @param current_time 当前时间（预留接口，当前未使用）
 * @return true=命中有效（重置miss计数），false=未命中（累加miss计数）
 *
 * 判定逻辑：health_change < threshold 即为命中
 *   - P4(打5号)阈值=-50：敌方5秒内掉血>50才算命中
 *   - P5(打4号)阈值=0：敌方只要掉血就算命中（保持自瞄）
 *   - P6(前哨站)阈值=-60：前哨站掉血>60才算命中
 *   - 基地区域阈值=-120：基地掉血>120才算命中
 *
 * 注意：health_change 的符号容易混淆——它是"旧血量-新血量"，所以敌方掉血时为正值
 *       但代码中 threshold 多为负值，所以判断条件 health_change < threshold 实际上
 *       要求 health_change 必须足够小（即敌方血量下降幅度必须足够大）
 */
bool ShootController::checkHealthChange(int health_change, int threshold, rclcpp::Time current_time) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (health_change < threshold) {
        // 命中有效：重置两个 miss 计数器
        consecutive_miss_count_for_disable_ = 0;
        consecutive_miss_count_for_swing_ = 0;
        return true;
    } else {
        // 未命中：累加计数
        consecutive_miss_count_for_disable_++;
        consecutive_miss_count_for_swing_++;
        return false;
    }
}

/**
 * @brief 判断是否应该禁用自瞄（AUTO → HARDCODE）
 * 仅在 AUTO 模式下检查，连续 miss 达到 disable 阈值（默认10次）时返回 true
 */
bool ShootController::shouldDisableAutoAim(rclcpp::Time) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (aim_mode_ != AimMode::AUTO) return false;
    return consecutive_miss_count_for_disable_ >= consecutive_miss_threshold_disable_;
}

/**
 * @brief 判断是否应该开始摆动扫描（HARDCODE → SWING）
 * 仅在 HARDCODE 模式下检查，连续 miss 达到 swing 阈值（默认3次）时返回 true
 *
 * 调用时序：先 disableAutoAim() 切到 HARDCODE，再调此方法判断是否进入 SWING
 */
bool ShootController::shouldStartSwinging(rclcpp::Time) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (aim_mode_ != AimMode::HARDCODE) return false;
    return consecutive_miss_count_for_swing_ >= consecutive_miss_threshold_swing_;
}

int ShootController::getConsecutiveMissCountForSwing() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return consecutive_miss_count_for_swing_;
}

/**
 * @brief 启动摆动扫描
 * 摆动机制：偏移量从 0 开始，在 ±swing_range 度范围内来回扫描。
 * CombatManager 将偏移叠加到路径点预设 yaw，因此不会进行全方位扫描。
 * 目的是在自瞄和硬编码都打不中时，通过扫描扩大搜索范围
 */
void ShootController::startSwinging(rclcpp::Time current_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    aim_mode_ = AimMode::SWING;
    swing_offset_yaw_ = 0.0;
    swing_direction_ = 1;
}

/**
 * @brief 停止摆动，回到 HARDCODE 模式
 * 当摆动期间检测到敌方血量变化（说明扫到了目标）时被 StateManager 调用
 */
void ShootController::stopSwinging(rclcpp::Time current_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    aim_mode_ = AimMode::HARDCODE;
    consecutive_miss_count_for_disable_ = 0;
    consecutive_miss_count_for_swing_ = 0;
    swing_offset_yaw_ = 0.0;
}

/**
 * @brief 更新摆动角度（每20ms由 swingLoop 调用）
 * @param dt 距上次调用的时间间隔（秒）
 * @return 当前摆动 yaw 偏移量（相对于路径点预设 yaw）
 *
 * 摆动轨迹：匀速三角波，在 ±swing_range 之间来回
 *   swing_offset_yaw += direction * speed * dt
 *   到达边界时反转方向
 */
double ShootController::updateSwingYaw(double dt) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (aim_mode_ != AimMode::SWING) return 0;

    swing_offset_yaw_ += swing_direction_ * swing_speed_ * dt;
    if (swing_offset_yaw_ > swing_range_) {
        swing_offset_yaw_ = swing_range_;
        swing_direction_ = -1;  // 到达右边界，向左摆
    } else if (swing_offset_yaw_ < -swing_range_) {
        swing_offset_yaw_ = -swing_range_;
        swing_direction_ = 1;  // 到达左边界，向右摆
    }

    return swing_offset_yaw_;
}

bool ShootController::isSwinging() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return aim_mode_ == AimMode::SWING;
}
