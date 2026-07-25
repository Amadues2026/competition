/**
 * @file waypoint_utils.h
 * @brief 路径点工具函数和常量声明 —— 敌方索引映射、血量阈值、路径点索引约定
 *
 * 本文件定义了路径点相关的全局常量和工具函数，
 * 是连接 PathStrategy、StateManager、CombatManager 的"索引字典"。
 *
 * 关键索引约定（固定规则，与 game_config.yaml 中的路径点顺序严格对应）：
 *   kFirstShootWaypointIndex = 4   第一个射击路径点（P5，索引4，0-based）
 *   kLoopStart = 6                 基地环路循环起点（P7，索引6）
 *   kBaseLoopStart = 6             同 kLoopStart
 *   kBaseWaypointEnd = 10          基地环路终点（P11，索引10）
 *
 * 敌方内部索引（0-5）与血量话题索引（0-11）的映射：
 *   healthTopicIndexToInternalIndex() 负责转换
 *
 * 注意：如果修改了路径点顺序或数量，这里的常量必须同步更新
 */

#ifndef COMPETITION_WAYPOINT_UTILS_H
#define COMPETITION_WAYPOINT_UTILS_H

#include <cstddef>
#include <vector>

struct Waypoint;

namespace waypoint_utils {

/** @brief 敌方单位数量（玩家+3号+4号+5号+前哨站+基地 = 6个） */
inline constexpr int kNumEnemies = 6;

/** @brief 快速移动模式下的初始路径点数量（P1-P4，前4个点全速奔跑） */
inline constexpr size_t kFastWaypointCount = 4;

// ── 路径点索引约定（0-based，与 path_strategy.cpp 对应） ──

/** @brief 第一个射击路径点索引（P5=索引4），之前的点为纯移动 */
inline constexpr size_t kFirstShootWaypointIndex = 4;
/** @brief 基地环路循环起点（P7=索引6） */
inline constexpr size_t kBaseLoopStart = 6;
/** @brief 基地环路终点（P11=索引10） */
inline constexpr size_t kBaseWaypointEnd = 10;

/** @brief 距离比较的浮点容差 */
inline constexpr double kDistanceEpsilon = 0.001;

// ── 血量变化阈值默认值（HP，可被 YAML 配置覆盖） ──

/** @brief P4 路径点血量变化阈值（-50，负数表示"敌方血量减少超过此值"） */
inline constexpr int HEALTH_CHANGE_THRESHOLD_P4 = -50;
/** @brief P5 路径点血量变化阈值（0，任何变化都触发） */
inline constexpr int HEALTH_CHANGE_THRESHOLD_P5 = 0;
/** @brief P6 路径点血量变化阈值（-60） */
inline constexpr int HEALTH_CHANGE_THRESHOLD_P6 = -60;
/** @brief 基地路径点血量变化阈值（-120，基地血量多需要更大变化量才触发） */
inline constexpr int HEALTH_CHANGE_THRESHOLD_BASE = -120;

/** @brief StateManager 使用的血量变化阈值值对象。 */
struct HealthChangeThresholdConfig {
    int p4 = HEALTH_CHANGE_THRESHOLD_P4;
    int p5 = HEALTH_CHANGE_THRESHOLD_P5;
    int p6 = HEALTH_CHANGE_THRESHOLD_P6;
    int base = HEALTH_CHANGE_THRESHOLD_BASE;
    int fallback = 50;
};

/** @brief 血量比例逃逸阈值（我方血量低于此比例时考虑撤退） */
inline constexpr double kHealthRatioEscape = 0.2;

// 距离门控参数已迁移至 YAML（game_config.yaml → RuntimeConfig.distance_gate_*）

/**
 * @brief 将血量话题全局索引转换为内部敌方索引（0-5）
 * @param topic_index 血量数组中的全局索引（0-11）
 * @param player_id   玩家ID（1=蓝方，2=红方）
 * @return 内部敌方索引（0-5），-1 表示不是敌方单位
 */
int healthTopicIndexToInternalIndex(int topic_index, int player_id);

/**
 * @brief 根据当前路径点索引确定应攻击的敌方单位内部索引（0-5）
 * @param player_id    玩家ID（保留参数，不依赖）
 * @param waypoint_idx 当前路径点索引
 * @return 敌方内部索引（0=敌方玩家, 2=4号, 3=5号, 4=前哨站, 5=基地）
 *
 * 返回值为内部索引，可直接用于血量数组偏移量计算，
 * 无需再调 healthTopicIndexToInternalIndex() 转换。
 */
int getEnemyIndex(int player_id, size_t waypoint_idx);

/**
 * @brief 查询当前路径点对应的血量变化阈值
 * @param config       已加载并验证的血量变化阈值
 * @param waypoint_idx 当前路径点索引
 * @return 阈值（HP）
 */
int getHealthChangeThreshold(const HealthChangeThresholdConfig& config, size_t waypoint_idx);

/**
 * @brief 查找路径序列中下一个带有射击任务的路径点
 * @param waypoints   完整路径点数组
 * @param current_idx 当前路径点索引
 * @return 下一个有射击任务的路径点索引（循环搜索）
 */
size_t findNextShootTaskIndex(const std::vector<Waypoint>& waypoints, size_t current_idx);

}  // namespace waypoint_utils

#endif
