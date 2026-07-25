/**
 * @file go_d2c2.cpp
 * @brief 程序主入口 —— 解析命令行参数，创建 GoNode 并启动 ROS2 事件循环
 *
 * 这是控制系统的可执行入口点（对应 CMakeLists.txt 中的 go_d2c2 target）。
 * 负责参数解析、错误处理、ROS2 生命周期管理。
 *
 * 使用方式：
 *   ros2 run go go_d2c2 <player_id> [test_waypoint_idx]
 *
 * 参数：
 *   player_id          必填，1=蓝方，2=红方
 *   test_waypoint_idx  可选，测试模式下的射击路径点编号（1-11）
 *                      -1=正常模式（所有射击点都生效）
 *                      指定后仅该点为射击点，用于单点调试
 *
 * 执行流程：
 *   1. rclcpp::init() 初始化 ROS2 通信
 *   2. 解析 argv 获取 player_id 和 test_waypoint_idx
 *   3. 创建 GoNode（内部完成所有模块初始化）
 *   4. MultiThreadedExecutor(2) 进入事件循环，5ms 控制循环独占一线程
 *   5. spin 返回后（如 ctrl+c 或 game_over_ 触发 shutdown）调用 rclcpp::shutdown()
 *
 * 错误处理：
 *   - 参数缺失/非法 → 打印用法提示后退出（return 1）
 *   - GoNode 构造异常 → 打印 Fatal 日志后关闭 ROS2（return 1）
 *   - spin 正常返回 → 正常退出（return 0）
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
