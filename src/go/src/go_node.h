/**
 * @file go_node.h
 * @brief ROS2 主节点声明 —— 系统初始化、模块组装、回调绑定、定时器驱动
 *
 * GoNode 是整个控制系统的入口和"胶水层"，负责：
 *   1. 加载配置文件，校验参数合法性
 *   2. 创建所有功能模块（DataManager, PathNavigator, ShootController 等）
 *   3. 组装三大控制器（MovementController, CombatManager, StateManager）
 *   4. 订阅模拟器话题，绑定回调函数
 *   5. 创建发布者，向模拟器发送控制指令
 *   6. 启动5个定时器，驱动各子系统的周期性循环
 *
 * 初始化顺序（有依赖关系，不能随意调换）：
 *   ConfigManager → 配置校验/模块专用值对象 → DataManager/PathNavigator/ShootController
 *   → YOLOv5Detector → PerformanceMonitor → 话题订阅/发布
 *   → MovementController → CombatManager → StateManager → 定时器
 *
 * 定时器：
 *   controlLoop()          5ms   运动控制（速度指令）
 *   shootLoop()            30ms  自瞄检测+弹道补偿（角度指令）
 *   positionAdjustLoop()   可配  射击期间位置保持
 *   healthCheckLoop()      1s    血量监控+自瞄降级决策
 *   swingLoop()            20ms  摆动扫描+射击指令发布
 */

#ifndef COMPETITION_GO_NODE_H
#define COMPETITION_GO_NODE_H

#include <atomic>
#include <memory>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "tdt_interface/msg/receive_data.hpp"
#include "tdt_interface/msg/send_data.hpp"

#include "runtime_config.h"
#include "shared_context.h"

// ── 前向声明（避免头文件循环依赖） ──
class DataManager;
class PathNavigator;
class IPathPlanningStrategy;
class ShootController;
class YOLOv5Detector;
class PerformanceMonitor;
class ConfigManager;
class Waypoint;
class MovementController;
class CombatManager;
class StateManager;
class Visualizer;

class GoNode : public rclcpp::Node {
private:
    // ── 共享上下文和运行时配置 ──
    SharedContext ctx_;     ///< 跨模块共享状态（底盘/云台/检测状态）
    RuntimeConfig config_;  ///< 从 YAML 加载的运行时参数

    // ── ROS2 订阅者（接收模拟器数据） ──
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr position_sub_;
    rclcpp::Subscription<tdt_interface::msg::ReceiveData>::SharedPtr real_angles_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr health_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr game_time_sub_;

    // ── ROS2 发布者（向模拟器发送控制指令） ──
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr speed_pub_;
    rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr angles_pub_;

    // ── 回调组（控制线程隔离，避免 5ms control 被其他回调阻塞） ──
    rclcpp::CallbackGroup::SharedPtr control_cb_group_;  ///< 仅 control_timer_ 使用
    rclcpp::CallbackGroup::SharedPtr other_cb_group_;    ///< 其余4个定时器 + 订阅共用

    // ── ROS2 定时器（驱动各子系统的周期性循环） ──
    rclcpp::TimerBase::SharedPtr control_timer_;          ///< 5ms - 运动控制（独立线程）
    rclcpp::TimerBase::SharedPtr shoot_timer_;            ///< 30ms - 自瞄检测
    rclcpp::TimerBase::SharedPtr position_adjust_timer_;  ///< 可配 - 位置保持
    rclcpp::TimerBase::SharedPtr health_check_timer_;     ///< 1s - 血量监控
    rclcpp::TimerBase::SharedPtr swing_timer_;            ///< 20ms - 摆动+射击

    // ── 路径点和测试模式（声明在引用者之前，保证析构时最后释放） ──
    std::vector<Waypoint> waypoints_;     ///< GoNode 唯一持有的可修改路径点数组
    int test_waypoint_idx_ = -1;          ///< 测试模式路径点编号（-1=正常）
    std::atomic<bool> game_over_{false};  ///< 游戏结束标志（基地被摧毁时置 true）

    // ── 基础功能模块 ──
    std::unique_ptr<DataManager> data_manager_;
    std::unique_ptr<PathNavigator> path_navigator_;
    std::unique_ptr<IPathPlanningStrategy> strategy_;
    std::unique_ptr<ShootController> shoot_controller_;
    std::unique_ptr<YOLOv5Detector> yolov5_detector_;
    std::unique_ptr<PerformanceMonitor> performance_monitor_;
    std::unique_ptr<ConfigManager> config_manager_;
    std::unique_ptr<Visualizer> visualizer_;  ///< 可视化模块（可选，YAML 控制）

    // ── 三大控制器（组装了基础模块） ──
    std::unique_ptr<MovementController> movement_;  ///< 底盘运动控制
    std::unique_ptr<CombatManager> combat_;         ///< 检测+射击控制
    std::unique_ptr<StateManager> state_mgr_;       ///< 状态+血量监控

    // ── 配置加载和校验 ──
    void loadConfigParameters();
    bool validateConfig();

    // ── ROS2 回调函数 ──
    void positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void anglesCallback(const tdt_interface::msg::ReceiveData::SharedPtr msg);
    void checkBaseDestroyed(size_t enemy_internal_index, int enemy_health);
    void healthCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg);
    void cameraCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void gameTimeCallback(const std_msgs::msg::Int32::SharedPtr msg);

    // ── 定时器回调 ──
    void controlLoop();
    void shootLoop();
    void positionAdjustLoop();
    void healthCheckLoop();
    void swingLoop();

public:
    GoNode(int player_id, int test_waypoint_idx = -1);
    ~GoNode() override;
};

#endif
