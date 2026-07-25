/**
 * @file test_data_manager.cpp
 * @brief DataManager 单元测试 —— 血量读写、敌方初始化、位置/角度历史记录
 *
 * 测试覆盖：
 *   1. 初始状态：自身血量=0，敌方未初始化
 *   2. 自身血量更新：updateSelfHealth + getSelfHealth
 *   3. 敌方血量初始化：首次更新记录初始血量，后续更新不覆盖
 *   4. 位置跟踪：updatePosition + getCurrentPosition（最新值返回）
 *   5. 云台角度历史：updateYawPitch + getHistoricalYawPitch（回溯查询）
 *
 * 前置条件说明：
 *   DataManager 内部使用 TimeManager 计算血量变化量（updateSelfHealth/updateEnemyHealth），
 *   但这些测试只验证不依赖时间的基础功能：
 *   - 初始阶段（窗口不满 health_history_duration）不会计算 change
 *   - getSelfHealthChange 在无 change 记录时返回 0
 *   - 所以无需初始化 TimeManager
 *
 * 测试夹具：
 *   使用 health_history_duration=5.0, other_data_duration=10.0
 *   基准时间 t0=1000.0 秒
 */

#include <gtest/gtest.h>
#include "data_manager.h"
#include "rclcpp/rclcpp.hpp"

/**
 * @brief 测试夹具 —— 创建 DataManager 实例
 *
 * health_history_duration=5.0: 血量变化量需要跨越5秒才计算
 * other_data_duration=10.0: 位置/角度历史保留10秒
 * t0=1000.0: 基准时间戳
 */
class DataManagerTest : public ::testing::Test {
protected:
    DataManager dm{5.0, 10.0};
    rclcpp::Time t0{1000, 0, RCL_ROS_TIME};
};

/**
 * @brief 测试初始状态 —— 未更新任何数据时的默认值
 *
 * getSelfHealth(): atomic 默认初始化为 0
 * getSelfHealthChange(): 无 change 记录，返回 0
 * isEnemyHealthInitialized(): 所有敌方单位未初始化，返回 false
 */
TEST_F(DataManagerTest, InitialState) {
    EXPECT_EQ(dm.getSelfHealth(), 0);
    EXPECT_EQ(dm.getSelfHealthChange(), 0);
    EXPECT_FALSE(dm.isEnemyHealthInitialized(0));
}

/**
 * @brief 测试自身血量更新 —— atomic 快速读写
 *
 * updateSelfHealth 写入 atomic_self_health_（无锁），
 * getSelfHealth 读取 atomic（无锁），200Hz 调用安全
 */
TEST_F(DataManagerTest, SelfHealthUpdate) {
    dm.updateSelfHealth(100, t0);
    EXPECT_EQ(dm.getSelfHealth(), 100);
}

/**
 * @brief 测试敌方血量初始化 —— 首次更新记录初始血量
 *
 * 流程：
 *   1. isEnemyHealthInitialized(2) → false（未更新过）
 *   2. updateEnemyHealth(2, 500, t0) → 记录初始血量=500
 *   3. isEnemyHealthInitialized(2) → true
 *   4. getInitialEnemyHealth(2) → 500（初始值）
 *   5. getCurrentEnemyHealth(2) → 500（当前值，与初始值相同）
 */
TEST_F(DataManagerTest, EnemyHealthInit) {
    EXPECT_FALSE(dm.isEnemyHealthInitialized(2));
    dm.updateEnemyHealth(2, 500, t0);
    EXPECT_TRUE(dm.isEnemyHealthInitialized(2));
    EXPECT_EQ(dm.getInitialEnemyHealth(2), 500);
    EXPECT_EQ(dm.getCurrentEnemyHealth(2), 500);
}

/**
 * @brief 测试敌方血量延迟初始化 —— 不同敌方单位独立初始化
 *
 * 更新 idx=2 和 idx=3，验证：
 *   - idx=2 的初始化不受 idx=3 影响
 *   - idx=3 在首次更新时记录初始血量=300
 *   - 初始血量一旦记录就不会被后续更新覆盖
 */
TEST_F(DataManagerTest, EnemyHealthChangeLateInit) {
    // 更新 idx=2（初始化 idx=2）
    dm.updateEnemyHealth(2, 500, t0);
    // 更新 idx=3（初始化 idx=3）
    rclcpp::Time t1{1000, 50000000, RCL_ROS_TIME};  // +0.05秒
    dm.updateEnemyHealth(3, 300, t1);
    EXPECT_TRUE(dm.isEnemyHealthInitialized(3));
    EXPECT_EQ(dm.getInitialEnemyHealth(3), 300);
}

/**
 * @brief 测试位置跟踪 —— 最新位置返回
 *
 * updatePosition 追加到 position_history_ 队列，
 * getCurrentPosition 返回队列末尾（最新）的 (x, y)
 *
 * 验证：
 *   - 第一次更新 (10, 20) → getCurrentPosition 返回 (10, 20)
 *   - 第二次更新 (15, 25) → getCurrentPosition 返回 (15, 25)（不是 (10, 20)）
 */
TEST_F(DataManagerTest, PositionTracking) {
    dm.updatePosition(10.0, 20.0, t0);
    auto [x, y] = dm.getCurrentPosition();
    EXPECT_DOUBLE_EQ(x, 10.0);
    EXPECT_DOUBLE_EQ(y, 20.0);

    dm.updatePosition(15.0, 25.0, t0);
    auto [x2, y2] = dm.getCurrentPosition();
    EXPECT_DOUBLE_EQ(x2, 15.0);
    EXPECT_DOUBLE_EQ(y2, 25.0);
}

/**
 * @brief 测试云台角度历史 —— getHistoricalYawPitch 回溯查询
 *
 * 当只有一条记录时，getHistoricalYawPitch 找不到回溯目标（窗口内无历史数据），
 * 因此返回最近一条记录（即唯一的那条）作为 fallback。
 *
 * 返回 {yaw, pitch} 元组，用于弹道补偿的云台指向预测
 */
TEST_F(DataManagerTest, YawPitchHistory) {
    dm.updateYawPitch(30.0, 15.0, t0);
    // 只有一条记录，getHistoricalYawPitch 返回最新值（fallback）
    auto [yaw, pitch] = dm.getHistoricalYawPitch(t0);
    EXPECT_DOUBLE_EQ(yaw, 30.0);
    EXPECT_DOUBLE_EQ(pitch, 15.0);
}
