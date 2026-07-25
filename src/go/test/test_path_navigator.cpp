/**
 * @file test_path_navigator.cpp
 * @brief PathNavigator 单元测试 —— 路径点切换、循环回退、卡住重试、超时跳过
 *
 * 测试覆盖：
 *   1. 初始状态：路径点索引从 0 开始
 *   2. moveToNext：路径点索引递增（0→1→2）
 *   3. 循环回退：到达末尾后从 kLoopStart(6) 重新开始
 *   4. 回退机制：markReturnToPrevious + moveToNext 回到上一个点
 *   5. 重试计数：incrementRetryCount 累加，超过 MAX_RETRY_COUNT(3) 触发超限
 *   6. 跳过当前点：skipCurrentPoint 重置计数并前进
 *   7. 强制导航：navigateToWaypoint 跳转到任意路径点
 *   8. 超时检测：isTimeout 在 move_timeout_(10s) 后返回 true
 *   9. 取消回退：cancelReturn 清除回退标志
 *  10. 重置计数：resetRetryCount 清零重试计数和超限标志
 *
 * 前置条件：
 *   - 测试注入 11 个路径点的 FixedPathStrategy 和默认 10s 超时
 */

#include <gtest/gtest.h>

#include <stdexcept>

#include "path_navigator.h"
#include "path_strategy.h"

/**
 * @brief 测试夹具 —— 创建蓝方 PathNavigator 实例
 *
 * 使用 PathNavigator 默认的 10.0 秒移动超时
 */
class PathNavigatorTest : public ::testing::Test {
protected:
    std::unique_ptr<FixedPathStrategy> strategy_;
    std::unique_ptr<PathNavigator> navigator_;

    void SetUp() override {
        strategy_ = std::make_unique<FixedPathStrategy>(11);
        navigator_ = std::make_unique<PathNavigator>(*strategy_);
    }
};

TEST(PathStrategyTest, GeneratesIndicesFromCountAndRejectsEmptyPath) {
    auto strategy = PathStrategyFactory::createStrategy(3);
    EXPECT_EQ(strategy->generatePathIndices(), (std::vector<size_t>{0, 1, 2}));
    EXPECT_THROW(PathStrategyFactory::createStrategy(0), std::invalid_argument);
}

/**
 * @brief 测试初始路径点索引为 0
 *
 * PathNavigator 构造后，current_target_index_=0，
 * getCurrentWaypointIndex() 应返回 path_indices_[0] = 0
 */
TEST_F(PathNavigatorTest, InitialWaypointIsValid) {
    size_t idx = navigator_->getCurrentWaypointIndex();
    EXPECT_EQ(idx, 0u);
}

/**
 * @brief 测试 moveToNext 递增路径点索引
 *
 * 调用一次 moveToNext 后，current_target_index_ 从 0 递增到 1，
 * getCurrentWaypointIndex() 应返回 path_indices_[1] = 1
 */
TEST_F(PathNavigatorTest, MoveToNextAdvancesIndex) {
    size_t first = navigator_->getCurrentWaypointIndex();
    navigator_->moveToNext(rclcpp::Time(0, 0, RCL_ROS_TIME));
    size_t second = navigator_->getCurrentWaypointIndex();
    EXPECT_EQ(first, 0u);
    EXPECT_EQ(second, 1u);
}

/**
 * @brief 测试路径循环回退 —— 到达末尾后从 kLoopStart(6) 重新开始
 *
 * 蓝方路径有 11 个点（索引 0-10），连续 moveToNext 11 次后：
 *   0→1→2→3→4→5→6→7→8→9→10→6（循环回 kLoopStart）
 *
 * kLoopStart=6 跳过了出发段的纯移动点（P1-P6），
 * 保证循环只在有射击任务的路径点（P7-P11 + P4-P6）之间进行
 */
TEST_F(PathNavigatorTest, LoopAfterEnd) {
    for (int i = 0; i < 11; ++i) {
        navigator_->moveToNext(rclcpp::Time(0, 0, RCL_ROS_TIME));
    }
    size_t idx = navigator_->getCurrentWaypointIndex();
    // 到达 P11 后循环到索引 6（P7）
    EXPECT_EQ(idx, 6u);
}

/**
 * @brief 测试回退机制 —— markReturnToPrevious + moveToNext 回到上一个点
 *
 * 流程：
 *   1. moveToNext 两次，当前在索引 2
 *   2. markReturnToPrevious() 设置 is_returning_=true, return_to_index_=1
 *   3. moveToNext() 检测到 is_returning_，直接跳到 return_to_index_=1（不递增）
 *   4. is_returning_ 自动清除
 */
TEST_F(PathNavigatorTest, MarkReturnToPrevious) {
    navigator_->moveToNext(rclcpp::Time(0, 0, RCL_ROS_TIME));
    navigator_->moveToNext(rclcpp::Time(0, 0, RCL_ROS_TIME));
    EXPECT_EQ(navigator_->getCurrentWaypointIndex(), 2u);

    navigator_->markReturnToPrevious();
    EXPECT_TRUE(navigator_->isReturning());

    navigator_->moveToNext(rclcpp::Time(0, 0, RCL_ROS_TIME));
    EXPECT_EQ(navigator_->getCurrentWaypointIndex(), 1u);
    EXPECT_FALSE(navigator_->isReturning());
}

/**
 * @brief 测试重试计数跟踪 —— 累加和超限检测
 *
 * MAX_RETRY_COUNT=3，调用 incrementRetryCount 3 次后：
 *   stuck_retry_count_=3, max_retries_exceeded_=true
 *   isMaxRetriesExceeded() 返回 true
 */
TEST_F(PathNavigatorTest, RetryCountTracking) {
    EXPECT_EQ(navigator_->getRetryCount(), 0);
    EXPECT_FALSE(navigator_->isMaxRetriesExceeded());

    navigator_->incrementRetryCount(rclcpp::Time(0, 0, RCL_ROS_TIME));
    EXPECT_EQ(navigator_->getRetryCount(), 1);
    EXPECT_FALSE(navigator_->isMaxRetriesExceeded());

    navigator_->incrementRetryCount(rclcpp::Time(0, 0, RCL_ROS_TIME));
    navigator_->incrementRetryCount(rclcpp::Time(0, 0, RCL_ROS_TIME));
    EXPECT_EQ(navigator_->getRetryCount(), 3);
    EXPECT_TRUE(navigator_->isMaxRetriesExceeded());
}

/**
 * @brief 测试跳过当前点 —— skipCurrentPoint 重置计数并前进
 *
 * 调用 skipCurrentPoint 后：
 *   1. stuck_retry_count_ 重置为 0
 *   2. max_retries_exceeded_ 重置为 false
 *   3. 调用 moveToNext 切换到下一个路径点
 */
TEST_F(PathNavigatorTest, SkipCurrentPoint) {
    navigator_->moveToNext(rclcpp::Time(0, 0, RCL_ROS_TIME));
    EXPECT_EQ(navigator_->getCurrentWaypointIndex(), 1u);

    navigator_->skipCurrentPoint(rclcpp::Time(0, 0, RCL_ROS_TIME));
    EXPECT_EQ(navigator_->getCurrentWaypointIndex(), 2u);
}

/**
 * @brief 测试强制导航 —— navigateToWaypoint 跳转到指定路径点
 *
 * 直接设置 current_target_index_ 匹配目标路径点索引
 */
TEST_F(PathNavigatorTest, NavigateToWaypoint) {
    navigator_->navigateToWaypoint(5, rclcpp::Time(0, 0, RCL_ROS_TIME));
    EXPECT_EQ(navigator_->getCurrentWaypointIndex(), 5u);
}

/**
 * @brief 测试超时检测 —— isTimeout 在 move_timeout_ 后返回 true
 *
 * moveToNext 会重置 start_time_to_current_ 为传入的时间。
 * 默认 move_timeout_=10.0 秒。
 *
 * 验证：
 *   - T=1 调用 moveToNext，start_time_=1
 *   - T=16 调用 isTimeout，时间差 15s > 10s → 超时
 */
TEST_F(PathNavigatorTest, IsTimeoutDetectsTimeout) {
    // moveToNext 重置 start_time_to_current_ 为 T=1
    navigator_->moveToNext(rclcpp::Time(1, 0, RCL_ROS_TIME));
    // 15秒后：15s > 10s（默认 move_timeout_）→ 超时
    EXPECT_TRUE(navigator_->isTimeout(rclcpp::Time(16, 0, RCL_ROS_TIME)));
}

TEST_F(PathNavigatorTest, UsesInjectedMoveTimeout) {
    PathNavigator navigator(*strategy_, 0.5);
    navigator.moveToNext(rclcpp::Time(1, 0, RCL_ROS_TIME));

    EXPECT_FALSE(navigator.isTimeout(rclcpp::Time(1, 400000000, RCL_ROS_TIME)));
    EXPECT_TRUE(navigator.isTimeout(rclcpp::Time(1, 600000000, RCL_ROS_TIME)));
}

/**
 * @brief 测试取消回退 —— cancelReturn 清除回退标志
 */
TEST_F(PathNavigatorTest, CancelReturn) {
    navigator_->moveToNext(rclcpp::Time(0, 0, RCL_ROS_TIME));
    navigator_->markReturnToPrevious();
    EXPECT_TRUE(navigator_->isReturning());

    navigator_->cancelReturn();
    EXPECT_FALSE(navigator_->isReturning());
}

/**
 * @brief 测试重置重试计数 —— resetRetryCount 清零计数和超限标志
 */
TEST_F(PathNavigatorTest, ResetRetryCount) {
    navigator_->incrementRetryCount(rclcpp::Time(0, 0, RCL_ROS_TIME));
    navigator_->incrementRetryCount(rclcpp::Time(0, 0, RCL_ROS_TIME));
    EXPECT_TRUE(navigator_->getRetryCount() > 0);

    navigator_->resetRetryCount();
    EXPECT_EQ(navigator_->getRetryCount(), 0);
    EXPECT_FALSE(navigator_->isMaxRetriesExceeded());
}
