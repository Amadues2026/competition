/**
 * @file go_d2c2.cpp
 * @brief ROS2 executable entry point for the competition controller.
 *
 * Run: ros2 run go S_d2c2 <player_id> [test_waypoint_idx] --ros-args
 *        -p engine_path:=/absolute/path/to/model.engine
 * Player 1 selects blue; player 2 selects red. The optional one-based waypoint
 * index restricts simulator firing tasks to one waypoint; -1 selects normal mode.
 * GoNode assembles modules and callbacks; a two-thread executor handles ROS2 work.
 */

#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>

#include "rclcpp/rclcpp.hpp"
#include "go_node.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <player_id> [test_waypoint_idx]" << std::endl;
        return 1;
    }

    int player_id = 0;
    try {
        player_id = std::stoi(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Invalid player_id argument: " << e.what() << std::endl;
        return 1;
    }

    int test_waypoint_idx = -1;
    if (argc >= 3) {
        try {
            test_waypoint_idx = std::stoi(argv[2]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid test_waypoint_idx argument: " << e.what() << std::endl;
            return 1;
        }
    }

    try {
        auto node = std::make_shared<GoNode>(player_id, test_waypoint_idx);
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        executor.spin();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
