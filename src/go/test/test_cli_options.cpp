#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "cli_options.h"

TEST(CliOptionsTest, AcceptsBothPlayersWithDefaultMode) {
    EXPECT_EQ(parseCliOptions({"S_d2c2", "1"}).player_id, 1);
    EXPECT_EQ(parseCliOptions({"S_d2c2", "2"}).player_id, 2);
    EXPECT_EQ(parseCliOptions({"S_d2c2", "1"}).test_waypoint_idx, -1);
}

TEST(CliOptionsTest, RosParametersAreRemovedBeforeParsingOptionalWaypoint) {
    const char* arguments[] = {"S_d2c2", "1", "--ros-args", "-p", "engine_path:=/tmp/model.engine"};
    const auto options = parseCliOptions(rclcpp::remove_ros_arguments(5, arguments));
    EXPECT_EQ(options.player_id, 1);
    EXPECT_EQ(options.test_waypoint_idx, -1);
}

TEST(CliOptionsTest, AcceptsWaypointBoundsAndExplicitNormalMode) {
    for (const int index : {-1, 1, 11}) {
        EXPECT_EQ(parseCliOptions({"S_d2c2", "1", std::to_string(index)}).test_waypoint_idx, index);
    }
}

TEST(CliOptionsTest, RejectsMissingOrExtraPositionalArguments) {
    EXPECT_THROW(parseCliOptions({"S_d2c2"}), std::invalid_argument);
    EXPECT_THROW(parseCliOptions({"S_d2c2", "1", "6", "unexpected"}), std::invalid_argument);
}

TEST(CliOptionsTest, RejectsInvalidPlayerAndPartialInteger) {
    for (const auto* player : {"0", "3", "-1", "1junk", "blue", ""}) {
        EXPECT_THROW(parseCliOptions({"S_d2c2", player}), std::invalid_argument);
    }
}

TEST(CliOptionsTest, RejectsInvalidWaypointAndPartialInteger) {
    for (const auto* waypoint : {"0", "12", "-2", "6junk", "--ros-args", ""}) {
        EXPECT_THROW(parseCliOptions({"S_d2c2", "1", waypoint}), std::invalid_argument);
    }
}
