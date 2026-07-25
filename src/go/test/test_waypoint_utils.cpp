/**
 * @file test_waypoint_utils.cpp
 * @brief waypoint_utils 工具函数单元测试 —— 索引映射、阈值查询、射击点搜索
 *
 * 测试覆盖：
 *   1. healthTopicIndexToInternalIndex：血量话题全局索引 → 内部敌方索引（蓝/红方）
 *   2. getEnemyIndex：路径点 → 敌方单位索引映射（蓝/红方）
 *   3. getHealthChangeThreshold：路径点 → 已注入的血量变化阈值
 *   4. findNextShootTaskIndex：查找下一个有射击任务的路径点
 *
 * 血量话题索引布局（固定，与 player_id 无关）：
 *   [0-5]  蓝方阵营（玩家/3号/4号/5号/前哨站/基地）
 *   [6-11] 红方阵营（玩家/3号/4号/5号/前哨站/基地）
 *
 * 内部敌方索引（0-5）：
 *   0=玩家, 1=3号, 2=4号, 3=5号, 4=前哨站, 5=基地
 *
 * 映射规则：
 *   蓝方视角(player_id=1)：敌方在话题索引 [6-11] → 内部索引 = topic_index - 6
 *   红方视角(player_id=2)：敌方在话题索引 [0-5]  → 内部索引 = topic_index
 */

#include <gtest/gtest.h>

#include "waypoint.h"
#include "waypoint_utils.h"

using namespace waypoint_utils;

/**
 * @brief 测试 healthTopicIndexToInternalIndex —— 蓝方视角
 *
 * 蓝方(player_id=1)的敌方在话题索引 [6-11]：
 *   topic[6] → internal[0]（敌方玩家）
 *   topic[7] → internal[1]（敌方3号）
 *   topic[8] → internal[2]（敌方4号）
 *   topic[9] → internal[3]（敌方5号）
 *   topic[10] → internal[4]（敌方前哨站）
 *   topic[11] → internal[5]（敌方基地）
 *
 * 己方话题 [0-5] 和越界话题 → 返回 -1
 */
TEST(WaypointUtilsTest, HealthTopicIndexToInternalIndex_Player1) {
    // Player1: topic[6..11] → internal[0..5]
    EXPECT_EQ(healthTopicIndexToInternalIndex(6, 1), 0);
    EXPECT_EQ(healthTopicIndexToInternalIndex(7, 1), 1);
    EXPECT_EQ(healthTopicIndexToInternalIndex(8, 1), 2);
    EXPECT_EQ(healthTopicIndexToInternalIndex(9, 1), 3);
    EXPECT_EQ(healthTopicIndexToInternalIndex(10, 1), 4);
    EXPECT_EQ(healthTopicIndexToInternalIndex(11, 1), 5);
    // 己方话题（0-5）应返回 -1（不是敌方单位）
    EXPECT_EQ(healthTopicIndexToInternalIndex(0, 1), -1);
    EXPECT_EQ(healthTopicIndexToInternalIndex(5, 1), -1);
    // 越界话题 → 返回 -1
    EXPECT_EQ(healthTopicIndexToInternalIndex(12, 1), -1);
}

/**
 * @brief 测试 healthTopicIndexToInternalIndex —— 红方视角
 *
 * 红方(player_id=2)的敌方在话题索引 [0-5]，直接映射
 * 己方话题 [6-11] → 返回 -1
 */
TEST(WaypointUtilsTest, HealthTopicIndexToInternalIndex_Player2) {
    // Player2: topic[0..5] → internal[0..5]（直接映射）
    EXPECT_EQ(healthTopicIndexToInternalIndex(0, 2), 0);
    EXPECT_EQ(healthTopicIndexToInternalIndex(1, 2), 1);
    EXPECT_EQ(healthTopicIndexToInternalIndex(2, 2), 2);
    EXPECT_EQ(healthTopicIndexToInternalIndex(3, 2), 3);
    EXPECT_EQ(healthTopicIndexToInternalIndex(4, 2), 4);
    EXPECT_EQ(healthTopicIndexToInternalIndex(5, 2), 5);
    // 敌方话题（6-11）应返回 -1
    EXPECT_EQ(healthTopicIndexToInternalIndex(6, 2), -1);
    EXPECT_EQ(healthTopicIndexToInternalIndex(11, 2), -1);
}

/**
 * @brief 测试 getEnemyIndex —— 蓝方视角
 *
 * 现在返回内部索引（0-5），红蓝方映射相同：
 *   P4(索引3) → 内部索引3（敌方5号）
 *   P5(索引4) → 内部索引2（敌方4号）
 *   P6(索引5) → 内部索引4（敌方前哨站）
 *   P7-P11(索引6-10) → 内部索引5（敌方基地）
 *   其他路径点 → 默认内部索引0（敌方玩家）
 */
TEST(WaypointUtilsTest, GetEnemyIndex_Player1) {
    EXPECT_EQ(getEnemyIndex(1, 3), 3);   // P4 → 内部索引3
    EXPECT_EQ(getEnemyIndex(1, 4), 2);   // P5 → 内部索引2
    EXPECT_EQ(getEnemyIndex(1, 5), 4);   // P6 → 内部索引4
    EXPECT_EQ(getEnemyIndex(1, 6), 5);   // P7 → 内部索引5
    EXPECT_EQ(getEnemyIndex(1, 7), 5);   // P8 → 内部索引5
    EXPECT_EQ(getEnemyIndex(1, 10), 5);  // P11 → 内部索引5
    EXPECT_EQ(getEnemyIndex(1, 0), 0);   // 其他 → 默认
}

/**
 * @brief 测试 getEnemyIndex —— 红方视角（红蓝方映射相同）
 */
TEST(WaypointUtilsTest, GetEnemyIndex_Player2) {
    EXPECT_EQ(getEnemyIndex(2, 3), 3);   // P4 → 内部索引3
    EXPECT_EQ(getEnemyIndex(2, 4), 2);   // P5 → 内部索引2
    EXPECT_EQ(getEnemyIndex(2, 5), 4);   // P6 → 内部索引4
    EXPECT_EQ(getEnemyIndex(2, 6), 5);   // P7 → 内部索引5
    EXPECT_EQ(getEnemyIndex(2, 10), 5);  // P11 → 内部索引5
    EXPECT_EQ(getEnemyIndex(2, 0), 0);   // 其他 → 默认
}

/**
 * @brief 测试 getHealthChangeThreshold —— 默认值对象
 *
 * 默认值对象使用代码默认常量：
 *   P4(索引3) → HEALTH_CHANGE_THRESHOLD_P4 = -50
 *   P5(索引4) → HEALTH_CHANGE_THRESHOLD_P5 = 0
 *   P6(索引5) → HEALTH_CHANGE_THRESHOLD_P6 = -60
 *   P7-P11(索引6-10) → HEALTH_CHANGE_THRESHOLD_BASE = -120
 *   其他路径点 → 默认值 50
 */
TEST(WaypointUtilsTest, GetHealthChangeThreshold_Fallback) {
    const HealthChangeThresholdConfig config;
    EXPECT_EQ(getHealthChangeThreshold(config, 3), HEALTH_CHANGE_THRESHOLD_P4);     // -50
    EXPECT_EQ(getHealthChangeThreshold(config, 4), HEALTH_CHANGE_THRESHOLD_P5);     // 0
    EXPECT_EQ(getHealthChangeThreshold(config, 5), HEALTH_CHANGE_THRESHOLD_P6);     // -60
    EXPECT_EQ(getHealthChangeThreshold(config, 6), HEALTH_CHANGE_THRESHOLD_BASE);   // -120
    EXPECT_EQ(getHealthChangeThreshold(config, 10), HEALTH_CHANGE_THRESHOLD_BASE);  // -120
    EXPECT_EQ(getHealthChangeThreshold(config, 99), 50);  // 未映射路径点 → 默认值 50
}

TEST(WaypointUtilsTest, GetHealthChangeThreshold_UsesInjectedValues) {
    const HealthChangeThresholdConfig config{-1, -2, -3, -4, -5};

    EXPECT_EQ(getHealthChangeThreshold(config, 3), -1);
    EXPECT_EQ(getHealthChangeThreshold(config, 4), -2);
    EXPECT_EQ(getHealthChangeThreshold(config, 5), -3);
    EXPECT_EQ(getHealthChangeThreshold(config, 6), -4);
    EXPECT_EQ(getHealthChangeThreshold(config, 99), -5);
}

/**
 * @brief 测试 findNextShootTaskIndex —— 查找下一个有射击任务的路径点
 *
 * 搜索逻辑：从 current_idx + 1 开始向后搜索 has_shoot_task=true 的路径点，
 * 到达末尾则循环回 kFirstShootWaypointIndex。
 *
 * 前置条件：路径点数组中至少有一个 has_shoot_task=true 的点
 * （否则会陷入无限循环，测试用例保证了这一点）
 *
 * 测试路径点布局：
 *   索引0: false, 索引1: true, 索引2: false, 索引3: true, 索引4: true
 *
 * 验证：
 *   从0往后找 → 1（第一个 true）
 *   从1往后找 → 3（跳过索引2的 false）
 *   从3往后找 → 4（紧邻的 true）
 *   从4往后找 → 越界 → 折回到 kFirstShootWaypointIndex(=4)
 */
TEST(WaypointUtilsTest, FindNextShootTaskIndex) {
    std::vector<Waypoint> waypoints = {
        {0, 0, 0, 0, false},  // 索引0: 无射击任务
        {0, 0, 0, 0, true},   // 索引1: 有射击任务
        {0, 0, 0, 0, false},  // 索引2: 无射击任务
        {0, 0, 0, 0, true},   // 索引3: 有射击任务
        {0, 0, 0, 0, true},   // 索引4: kFirstShootWaypointIndex，有射击任务（防止无限循环）
    };
    // 从索引0往后找 → 找到索引1
    EXPECT_EQ(findNextShootTaskIndex(waypoints, 0), 1u);
    // 从索引1往后找 → 跳过索引2，找到索引3
    EXPECT_EQ(findNextShootTaskIndex(waypoints, 1), 3u);
    // 从索引3往后找 → 找到索引4（紧邻的 true）
    EXPECT_EQ(findNextShootTaskIndex(waypoints, 3), 4u);
    // 从索引4往后找 → 越界 → 折回到 kFirstShootWaypointIndex(=4)
    EXPECT_EQ(findNextShootTaskIndex(waypoints, 4), 4u);
}
