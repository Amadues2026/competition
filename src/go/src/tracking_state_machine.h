/**
 * @file tracking_state_machine.h
 * @brief 目标跟踪状态机声明 —— 管理 LOST/DETECTING/TRACKING/TEMP_LOST 四状态转换
 *
 * 状态机用于防止单帧误检导致的虚假跟踪，核心逻辑：
 *   LOST → (首次检测) → DETECTING → (连续N帧确认) → TRACKING
 *   TRACKING → (短暂丢失) → TEMP_LOST → (找回) → TRACKING
 *   任意非LOST + NIS过高 → 强制 LOST（滤波器发散）
 *
 * 默认参数：min_detect_count=3, max_temp_lost_count=15, nis_failure_max=0.3
 *
 * 使用场景：
 *   CombatManager 中的 tracking_sm_ 实例，每帧 shootLoop() 调用 update()，
 *   只有 state()==TRACKING 时才输出弹道补偿结果。
 */

#ifndef COMPETITION_TRACKING_STATE_MACHINE_H
#define COMPETITION_TRACKING_STATE_MACHINE_H

/**
 * @brief 跟踪状态枚举
 *   LOST      - 未跟踪任何目标
 *   DETECTING - 检测到目标，等待连续确认
 *   TRACKING  - 稳定跟踪中（可开火）
 *   TEMP_LOST - 目标短暂丢失（等待找回）
 */
enum class TrackingState { LOST, DETECTING, TRACKING, TEMP_LOST };

class TrackingStateMachine {
public:
    /**
     * @brief 构造函数
     * @param min_detect_count      DETECTING→TRACKING 所需最少连续检测帧数
     * @param max_temp_lost_count   TEMP_LOST 下允许的最大未检测帧数
     * @param nis_failure_max       NIS 失败率阈值（超过则强制 LOST）
     */
    TrackingStateMachine(int min_detect_count = 3,
                         int max_temp_lost_count = 15,
                         double nis_failure_max = 0.3);

    /** @brief 状态机更新（每帧调用一次） */
    void update(bool target_detected, double nis_failure_rate);

    /** @brief 获取当前状态 */
    TrackingState state() const { return state_; }

    /**
     * @brief 判断是否应执行跟踪（DETECTING 或 TRACKING 状态）
     *
     * DETECTING 状态下也可以执行跟踪（但可能不输出弹道补偿），
     * 用于提前开始云台指向调整，减少 TRACKING 后的响应延迟。
     */
    bool shouldTrack() const {
        return state_ == TrackingState::DETECTING || state_ == TrackingState::TRACKING;
    }

    /** @brief 重置到 LOST 状态（切换目标时调用） */
    void reset();

private:
    TrackingState state_;           ///< 当前跟踪状态
    int min_detect_count_;          ///< 确认检测所需的最少帧数
    int max_temp_lost_count_;       ///< 临时丢失的最大容忍帧数
    double nis_failure_max_;        ///< NIS 失败率阈值
    int detect_count_;              ///< DETECTING 状态下的连续检测计数
    int temp_lost_count_;           ///< TEMP_LOST 状态下的连续未检测计数
};

#endif
