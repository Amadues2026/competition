#include <gtest/gtest.h>

#include "config_manager.h"
#include "rolling_median_filter.h"

TEST(ConfigManagerTest, LoadsWaypointsFromYaml) {
    ConfigManager manager;
    ASSERT_TRUE(manager.loadFromFile(TEST_CONFIG_PATH));

    const auto& config = manager.getConfig();
    ASSERT_EQ(config.player1_waypoints.size(), 11u);
    ASSERT_EQ(config.player2_waypoints.size(), 11u);

    EXPECT_DOUBLE_EQ(config.player1_waypoints.front().x, 15.9);
    EXPECT_DOUBLE_EQ(config.player2_waypoints.front().x, -15.9);
    EXPECT_FALSE(config.player1_waypoints.front().has_shoot_task);
    EXPECT_TRUE(config.player1_waypoints.at(3).has_shoot_task);
    EXPECT_DOUBLE_EQ(config.ekf.q_adaptive_max_scale, 10.0);
    EXPECT_DOUBLE_EQ(config.nis.chi2_upper, 7.815);
    EXPECT_EQ(config.timer.visualization, 67);
}

TEST(RollingMedianFilterTest, RejectsNonPositiveWindow) {
    EXPECT_THROW(RollingMedianFilter(0), std::invalid_argument);
    EXPECT_THROW(RollingMedianFilter(-1), std::invalid_argument);
}
