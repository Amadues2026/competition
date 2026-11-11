#ifndef COMPETITION_CLI_OPTIONS_H
#define COMPETITION_CLI_OPTIONS_H

#include <stdexcept>
#include <string>
#include <vector>

struct CliOptions {
    int player_id;
    int test_waypoint_idx = -1;
};

// Accept arguments after rclcpp::remove_ros_arguments(), including argv[0].
inline CliOptions parseCliOptions(const std::vector<std::string>& arguments) {
    if (arguments.size() < 2 || arguments.size() > 3) {
        throw std::invalid_argument("Expected <player_id> [test_waypoint_idx]");
    }
    const auto parse_integer = [](const std::string& text) {
        size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("Expected an integer, received: " + text);
        }
        return value;
    };
    CliOptions options{parse_integer(arguments[1]), -1};
    if (options.player_id != 1 && options.player_id != 2) {
        throw std::invalid_argument("player_id must be 1 (blue) or 2 (red)");
    }
    if (arguments.size() == 3) {
        options.test_waypoint_idx = parse_integer(arguments[2]);
        if (options.test_waypoint_idx != -1 &&
            (options.test_waypoint_idx < 1 || options.test_waypoint_idx > 11)) {
            throw std::invalid_argument("test_waypoint_idx must be -1 or an index from 1 to 11");
        }
    }
    return options;
}

#endif
