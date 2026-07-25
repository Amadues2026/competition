/**
 * @file test_time_manager.cpp
 * @brief TimeManager 单元测试 —— 时间差计算、超时判断、历史记录过期统计、初始化
 *
 * 测试覆盖：
 *   1. timeSince：两个时间戳之间的差值计算（正/负/零）
 *   2. isTimeout：超时判断的边界条件（未超时/刚好超时/超过阈值）
 *   3. countExpiredRecords：历史记录过期统计（空队列/单元素/部分过期/全部过期）
 *   4. initialize + updateGameTime：初始化基准时间和游戏时间映射
 *
 * 注意：TimeManager 是静态类，测试间会共享状态。
 *   - InitializeAndGameTime 测试会修改全局状态（initialize + updateGameTime）
 *   - 其他测试只使用 timeSince/isTimeout/countExpiredRecords，不依赖全局状态
 *
 * 时间单位：
 *   rclcpp::Time 的构造参数为 (秒, 纳秒, 时间系统)
 *   例如 rclcpp::Time(100, 0, RCL_ROS_TIME) 表示 ROS 时间 100.0 秒
 */

#include <gtest/gtest.h>
#include "time_manager.h"

/**
 * @brief 测试 timeSince —— 计算两个时间戳之间的差值
 *
 * timeSince(t0, t1) = (t1 - t0).seconds()
 *   - t1 > t0 → 正值（t1 比 t0 晚 5 秒）
 *   - t1 < t0 → 负值（t1 比 t0 早 5 秒）
 *   - t1 == t0 → 零
 */
TEST(TimeManagerTest, TimeSince) {
    rclcpp::Time t0(100, 0, RCL_ROS_TIME);
    rclcpp::Time t1(105, 0, RCL_ROS_TIME);
    EXPECT_DOUBLE_EQ(TimeManager::timeSince(t0, t1), 5.0);
    EXPECT_DOUBLE_EQ(TimeManager::timeSince(t1, t0), -5.0);
    EXPECT_DOUBLE_EQ(TimeManager::timeSince(t0, t0), 0.0);
}

/**
 * @brief 测试 isTimeout —— 超时判断的边界条件
 *
 * isTimeout(start, current, timeout) = timeSince(start, current) > timeout
 *   - 3s < 10s → 未超时
 *   - 11s > 10s → 超时
 *   - 11s > 5s  → 超时
 *
 * 注意：使用严格大于（>），刚好等于阈值时不算超时
 */
TEST(TimeManagerTest, IsTimeout) {
    rclcpp::Time start(0, 0, RCL_ROS_TIME);
    rclcpp::Time soon(3, 0, RCL_ROS_TIME);
    rclcpp::Time late(11, 0, RCL_ROS_TIME);
    EXPECT_FALSE(TimeManager::isTimeout(start, soon, 10.0));
    EXPECT_TRUE(TimeManager::isTimeout(start, late, 10.0));
    EXPECT_TRUE(TimeManager::isTimeout(start, late, 5.0));
}

/**
 * @brief 测试 countExpiredRecords —— 统计时间戳队列中的过期记录数
 *
 * 假设：timestamps 按时间单调递增（最早→最晚）
 * 规则：timeSince(ts, now) > max_duration → 过期
 * 边界：至少保留1条记录（全部过期时返回 size()-1）
 *
 * 测试场景：
 *   - 空队列 → 0
 *   - 1 个元素（size ≤ 1）→ 0（保底保留）
 *   - 2 个元素，第一个过期（20s > 10s）→ 1
 *   - 2 个元素，都在窗口内 → 0
 *   - 2 个元素，全部过期（50s/40s > 10s）→ 2
 */
TEST(TimeManagerTest, CountExpiredRecords) {
    rclcpp::Time now(100, 0, RCL_ROS_TIME);
    std::deque<rclcpp::Time> timestamps;

    // 空队列 → 返回 0
    EXPECT_EQ(TimeManager::countExpiredRecords(now, timestamps, 10.0), 0u);

    // 1 个元素（size ≤ 1 → 返回 0，保底保留）
    timestamps.push_back(rclcpp::Time(80, 0, RCL_ROS_TIME));
    EXPECT_EQ(TimeManager::countExpiredRecords(now, timestamps, 10.0), 0u);

    // 2 个元素，第一个过期（20秒前 > 10秒窗口）
    timestamps.push_back(rclcpp::Time(95, 0, RCL_ROS_TIME));
    EXPECT_EQ(TimeManager::countExpiredRecords(now, timestamps, 10.0), 1u);

    // 2 个元素，都在窗口内（5秒前和3秒前 < 10秒）
    std::deque<rclcpp::Time> recent;
    recent.push_back(rclcpp::Time(95, 0, RCL_ROS_TIME));
    recent.push_back(rclcpp::Time(97, 0, RCL_ROS_TIME));
    EXPECT_EQ(TimeManager::countExpiredRecords(now, recent, 10.0), 0u);

    // 2 个元素，全部过期（50秒前和40秒前 > 10秒）
    std::deque<rclcpp::Time> all_old;
    all_old.push_back(rclcpp::Time(50, 0, RCL_ROS_TIME));
    all_old.push_back(rclcpp::Time(60, 0, RCL_ROS_TIME));
    EXPECT_EQ(TimeManager::countExpiredRecords(now, all_old, 10.0), 2u);
}

/**
 * @brief 测试初始化和游戏时间映射
 *
 * getCurrentTime() = game_start_time_ + current_game_time_
 *
 * 验证：
 *   1. initialize(1000) 设置基准时间
 *   2. getCurrentTime() 返回 1000.0（基准 + 游戏时间0）
 *   3. updateGameTime(30) 设置游戏时间为30秒
 *   4. getCurrentTime() 返回 1030.0（1000 + 30）
 */
TEST(TimeManagerTest, InitializeAndGameTime) {
    // getCurrentTime 返回 game_start_time_ + current_game_time_
    TimeManager::initialize(rclcpp::Time(1000, 0, RCL_ROS_TIME));
    rclcpp::Time t = TimeManager::getCurrentTime();
    EXPECT_EQ(t.seconds(), 1000.0);

    TimeManager::updateGameTime(30);
    t = TimeManager::getCurrentTime();
    EXPECT_EQ(t.seconds(), 1030.0);
}
