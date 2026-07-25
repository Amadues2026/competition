/**
 * @file tracking_state_machine.cpp
 * @brief 目标跟踪状态机 —— 管理检测/跟踪/丢失状态的转换，防止误跟踪
 *
 * 状态机包含 4 个状态，防止单帧误检导致的虚假跟踪：
 *
 *   LOST ──(检测到)──→ DETECTING ──(连续N帧)──→ TRACKING
 *     ↑                    │                         │
 *     │                    │(未检测到)                │(未检测到)
 *     │                    ↓                         ↓
 *     └─────────────── LOST              TEMP_LOST ──(找回)──→ TRACKING
 *                                              │
 *                                              │(连续M帧未找回)
 *                                              ↓
 *                                            LOST
 *
 * 全局异常转换：
 *   任意非 LOST 状态 + NIS 失败率 > 阈值 → 强制 LOST
 *   （说明滤波器发散，当前跟踪不可信）
 *
 * 各状态含义：
 *   LOST      - 未跟踪任何目标，等待首次检测
 *   DETECTING - 刚检测到目标，等待连续确认（防止单帧误检）
 *   TRACKING  - 稳定跟踪中，可以执行射击
 *   TEMP_LOST - 目标短暂丢失（遮挡/出视野），保持跟踪状态等待找回
 *
 * 使用场景：
 *   YOLOv5Detector 在 calculateBallisticCompensation() 中每帧调用 update()，
 *   只有状态为 TRACKING 时才输出弹道补偿结果。
 *   TEMP_LOST 状态下可以继续使用 EKF 预测值（目标位置外推）。
 */

#include "tracking_state_machine.h"

/**
 * @brief 构造函数
 * @param min_detect_count      从 DETECTING → TRACKING 所需的最少连续检测帧数
 * @param max_temp_lost_count   TEMP_LOST 状态下允许的最大未检测帧数，超过则回退到 LOST
 * @param nis_failure_max       NIS 失败率阈值，超过则强制 LOST（滤波器发散检测）
 *
 * 参数调参建议：
 *   min_detect_count 太小 → 容易被单帧误检触发跟踪
 *   min_detect_count 太大 → 真实目标也被延迟跟踪（浪费开火时间）
 *   max_temp_lost_count 太小 → 短暂遮挡就丢失目标（频繁重新初始化 EKF）
 *   max_temp_lost_count 太大 → 目标已消失但仍跟踪（浪费弹药）
 */
TrackingStateMachine::TrackingStateMachine(int min_detect_count,
                                           int max_temp_lost_count,
                                           double nis_failure_max)
    : state_(TrackingState::LOST),
      min_detect_count_(min_detect_count),
      max_temp_lost_count_(max_temp_lost_count),
      nis_failure_max_(nis_failure_max),
      detect_count_(0), temp_lost_count_(0) {}

/**
 * @brief 状态机更新（每帧调用一次）
 * @param target_detected 本帧是否检测到目标（bool）
 * @param nis_failure_rate EKF 的 NIS 失败率（0.0~1.0）
 *
 * 转换优先级：
 *   1. NIS 检查（最高优先级）：如果滤波器发散，无论当前状态如何都强制回 LOST
 *   2. 状态内转换：根据 target_detected 在当前状态内做转换
 *
 * 状态转换表：
 *   LOST:      detected → DETECTING (detect_count=1)
 *   DETECTING: detected → detect_count++, if >= min → TRACKING
 *              !detected → LOST (detect_count=0)
 *   TRACKING:  !detected → TEMP_LOST (temp_lost_count=1)
 *   TEMP_LOST: detected → TRACKING
 *              !detected → temp_lost_count++, if > max → LOST
 */
void TrackingStateMachine::update(bool target_detected, double nis_failure_rate) {
    // 全局检查：NIS 失败率过高说明滤波器发散，强制丢失
    if (state_ != TrackingState::LOST && nis_failure_rate > nis_failure_max_) {
        state_ = TrackingState::LOST;
        detect_count_ = 0;
        temp_lost_count_ = 0;
        return;
    }

    switch (state_) {
    case TrackingState::LOST:
        // 等待首次检测到目标
        if (target_detected) {
            state_ = TrackingState::DETECTING;
            detect_count_ = 1;
        }
        break;

    case TrackingState::DETECTING:
        if (target_detected) {
            // 连续检测确认中
            if (++detect_count_ >= min_detect_count_)
                state_ = TrackingState::TRACKING;
        } else {
            // 确认过程中丢失 → 回退到 LOST（单帧误检被过滤掉）
            state_ = TrackingState::LOST;
            detect_count_ = 0;
        }
        break;

    case TrackingState::TRACKING:
        if (!target_detected) {
            // 目标短暂丢失（遮挡/出视野），进入临时丢失状态
            state_ = TrackingState::TEMP_LOST;
            temp_lost_count_ = 1;
        }
        break;

    case TrackingState::TEMP_LOST:
        if (target_detected) {
            // 找回目标 → 恢复跟踪（无需重新确认）
            state_ = TrackingState::TRACKING;
        } else if (++temp_lost_count_ > max_temp_lost_count_) {
            // 长期未找回 → 彻底丢失（需要重新确认）
            state_ = TrackingState::LOST;
            temp_lost_count_ = 0;
        }
        break;
    }
}

/**
 * @brief 重置状态机到初始 LOST 状态
 *
 * 重置时机：
 *   - 切换射击目标时（避免旧目标的跟踪状态影响新目标）
 *   - 检测到目标跳变（如中心距超过100像素，可能是不同目标）
 */
void TrackingStateMachine::reset() {
    state_ = TrackingState::LOST;
    detect_count_ = 0;
    temp_lost_count_ = 0;
}
