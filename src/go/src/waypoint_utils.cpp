/**
 * @file waypoint_utils.cpp
 * @brief 路径点工具函数 —— 敌方目标索引映射、血量阈值查询、下一射击点查找
 *
 * 本文件提供路径点相关的静态工具函数，被 StateManager、CombatManager、PathNavigator 等多处调用。
 * 核心职责：
 *   1. 根据当前路径点确定应该攻击哪个敌方单位（getEnemyIndex）
 *   2. 根据当前路径点查询对应的血量变化阈值（getHealthChangeThreshold）
 *   3. 查找路径序列中下一个带有射击任务的路径点（findNextShootTaskIndex）
 *   4. 将血量话题中的全局索引转换为内部 0-5 的敌方索引（healthTopicIndexToInternalIndex）
 *
 * 路径点与敌方目标的映射关系（蓝方 player_id=1 为例）：
 *   P4(索引3)  → 攻击敌方4号（内部索引3, 话题索引9）
 *   P5(索引4)  → 攻击敌方3号（内部索引2, 话题索引8）
 *   P6(索引5)  → 攻击敌方前哨站（内部索引4, 话题索引10）
 *   P7-P11(索引6-10) → 攻击敌方基地（内部索引5, 话题索引11）
 *   其他路径点（出发段） → 默认攻击内部索引0（敌方玩家）
 *
 * 注意：索引映射是固定比赛规则，与 game_config.yaml 的路径点顺序严格对应。
 * 如果修改了路径点顺序或数量，这里的映射必须同步更新。
 */

#include "waypoint_utils.h"

#include "waypoint.h"

namespace waypoint_utils {

/**
 * @brief 根据当前路径点索引返回应攻击的敌方单位内部索引（0-5）
 * @param player_id  玩家ID（保留参数，不再依赖）
 * @param waypoint_idx 当前路径点在 waypoints_ 中的原始索引
 * @return 敌方内部索引（0=敌方玩家, 1-4=3/4/5号及前哨站, 5=敌方基地）
 *
 * 映射逻辑（红蓝方相同）：
 *   路径点3(P4) → 敌方5号（索引3）  — 中路射击位
 *   路径点4(P5) → 敌方4号（索引2）  — 侧面射击位
 *   路径点5(P6) → 敌方前哨站（索引4）— 前哨站射击位
 *   路径点6-10(P7-P11) → 敌方基地（索引5）— 基地环路
 *   其他 → 默认打敌方玩家（索引0）
 *
 * 返回的是内部索引（0-5），可直接用于血量数组偏移量计算，
 * 无需再调 healthTopicIndexToInternalIndex() 转换。
 */
int getEnemyIndex(int /*player_id*/, size_t waypoint_idx) {
    switch (waypoint_idx) {
        case 3:
            return 3;
        case 4:
            return 2;
        case 5:
            return 4;
        case kBaseLoopStart:
        case 7:
        case 8:
        case 9:
        case kBaseWaypointEnd:
            return 5;
        default:
            return 0;
    }
}

/**
 * @brief 根据当前路径点查询血量变化阈值（HP）
 * @param config 已由组装层加载和验证的阈值
 * @param waypoint_idx 当前路径点索引
 * @return 阈值（HP），单位为血量变化量的绝对值
 *
 * 使用场景：StateManager 在 healthCheckLoop() 中检查敌方血量变化，
 * 如果 |change| > 阈值，说明该敌方单位正在被攻击，优先集火。
 * 不同路径点使用不同阈值是因为各目标的血量和战场重要性不同。
 */
int getHealthChangeThreshold(const HealthChangeThresholdConfig& config, size_t waypoint_idx) {
    switch (waypoint_idx) {
        case 3:
            return config.p4;
        case 4:
            return config.p5;
        case 5:
            return config.p6;
        case kBaseLoopStart:
        case 7:
        case 8:
        case 9:
        case kBaseWaypointEnd:
            return config.base;
        default:
            return config.fallback;
    }
}

/**
 * @brief 查找路径序列中下一个带有射击任务的路径点索引
 * @param waypoints     完整路径点数组
 * @param current_idx   当前路径点索引
 * @return 下一个有射击任务的路径点索引
 *
 * 搜索逻辑：
 *   1. 从 current_idx + 1 开始向后搜索
 *   2. 找到末尾则循环回 kFirstShootWaypointIndex（跳过出发段的纯移动点）
 *   3. 找到 has_shoot_task == true 的点即返回
 *
 * 保证：至少会返回自身（因为当前点就是有射击任务的点）
 * 用途：StateManager 的自瞄降级逻辑需要知道"下一个要打谁"来决定是否继续当前射击
 */
size_t findNextShootTaskIndex(const std::vector<Waypoint>& waypoints, size_t current_idx) {
    size_t next_idx = current_idx + 1;
    if (next_idx >= waypoints.size()) {
        next_idx = kFirstShootWaypointIndex;
    }
    while (next_idx < waypoints.size() && !waypoints[next_idx].has_shoot_task) {
        next_idx++;
        if (next_idx >= waypoints.size()) {
            next_idx = kFirstShootWaypointIndex;
        }
    }
    return next_idx;
}

/**
 * @brief 将血量话题中的全局索引转换为内部敌方索引（0-5）
 * @param topic_index  血量数组中的全局索引（0-11）
 * @param player_id    玩家ID（1=蓝方，2=红方）
 * @return 内部敌方索引（0-5），-1 表示索引无效（不是敌方单位）
 *
 * 血量话题索引布局（固定，与 player_id 无关）：
 *   [0-5]  蓝方阵营（玩家/3号/4号/5号/前哨站/基地）
 *   [6-11] 红方阵营（玩家/3号/4号/5号/前哨站/基地）
 *
 * 转换规则：
 *   蓝方视角(player_id=1)：敌方在 [6-11] → 内部索引 = topic_index - 6
 *   红方视角(player_id=2)：敌方在 [0-5]  → 内部索引 = topic_index
 *
 * 注意：返回 -1 的情况包括：
 *   - topic_index 指向的是我方单位（不是敌方）
 *   - topic_index 越界
 *   - player_id 无效
 */
int healthTopicIndexToInternalIndex(int topic_index, int player_id) {
    // 蓝方视角：敌方话题索引 [6,11] → 内部索引 [0,5]
    if (player_id == 1) {
        if (topic_index >= 6 && topic_index <= 11) {
            return topic_index - 6;
        }
    } else {
        // 红方视角：敌方话题索引 [0,5] → 内部索引 [0,5]（直接映射）
        if (topic_index >= 0 && topic_index <= 5) {
            return topic_index;
        }
    }
    return -1;
}

}  // namespace waypoint_utils
