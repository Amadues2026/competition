#include <gtest/gtest.h>

#include "shoot_controller.h"

TEST(ShootControllerTest, SwingOffsetStaysWithinConfiguredRange) {
    ShootControllerConfig config;
    config.swing_range = 2.0;
    config.swing_speed = 20.0;
    ShootController controller(config);
    controller.startSwinging(rclcpp::Time(0, 0, RCL_ROS_TIME));

    EXPECT_DOUBLE_EQ(controller.updateSwingYaw(0.0), 0.0);
    EXPECT_DOUBLE_EQ(controller.updateSwingYaw(0.05), 1.0);
    EXPECT_DOUBLE_EQ(controller.updateSwingYaw(0.05), 2.0);
    EXPECT_DOUBLE_EQ(controller.updateSwingYaw(0.05), 2.0);
    EXPECT_DOUBLE_EQ(controller.updateSwingYaw(0.05), 1.0);
    for (int i = 0; i < 1000; ++i) {
        const double offset = controller.updateSwingYaw(0.02);
        EXPECT_GE(offset, -2.0);
        EXPECT_LE(offset, 2.0);
    }
}

TEST(ShootControllerTest, UsesInjectedMissThresholds) {
    ShootControllerConfig config;
    config.consecutive_miss_disable = 2;
    config.consecutive_miss_swing = 1;
    ShootController controller(config);
    const rclcpp::Time now(0, 0, RCL_ROS_TIME);

    controller.startShooting(100, now);
    EXPECT_FALSE(controller.checkHealthChange(0, -50, now));
    EXPECT_FALSE(controller.shouldDisableAutoAim(now));
    EXPECT_FALSE(controller.checkHealthChange(0, -50, now));
    EXPECT_TRUE(controller.shouldDisableAutoAim(now));

    controller.disableAutoAim();
    EXPECT_FALSE(controller.shouldStartSwinging(now));
    EXPECT_FALSE(controller.checkHealthChange(0, -50, now));
    EXPECT_TRUE(controller.shouldStartSwinging(now));
}
