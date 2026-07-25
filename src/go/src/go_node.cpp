/**
 * @file go_node.cpp
 * @brief ROS2 主节点 —— 系统初始化、模块组装、回调绑定、定时器驱动
 *
 * 本文件是整个控制系统的入口和"胶水层"，负责：
 *
 * 1. 加载配置文件，校验参数合法性
 * 2. 创建所有功能模块（DataManager, PathNavigator, ShootController, YOLOv5Detector 等）
 * 3. 组装三大控制器（MovementController, CombatManager, StateManager）
 * 4. 订阅模拟器话题，绑定回调函数
 * 5. 创建发布者，向模拟器发送控制指令
 * 6. 启动5个定时器，驱动各子系统的周期性循环
 *
 * 初始化顺序（有依赖关系，不能随意调换）：
 *   ConfigManager → 配置校验/模块专用值对象 → DataManager/PathNavigator/ShootController
 *   → YOLOv5Detector → PerformanceMonitor → 话题订阅/发布
 *   → MovementController → CombatManager → StateManager → 定时器
 *
 * 话题映射（player_id 决定订阅/发布哪组话题）：
 *   Player1(蓝方) → /position_player_1, /camera_image_player_1, ...
 *   Player2(红方) → /position_player_2, /camera_image_player_2, ...
 *   双方共享 → /game_healths, /game_time
 *
 * 定时器：
 *   controlLoop()          5ms   运动控制（速度指令）
 *   shootLoop()            30ms  自瞄检测+弹道补偿（角度指令）
 *   positionAdjustLoop()   可配  射击期间位置保持
 *   healthCheckLoop()      1s    血量监控+自瞄降级决策
 *   swingLoop()            20ms  摆动扫描+射击指令发布
 */

#include "go_node.h"

#include <cmath>
#include <functional>
#include <type_traits>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "cv_bridge/cv_bridge.h"

#include "combat_manager.h"
#include "config_manager.h"
#include "data_manager.h"
#include "movement_controller.h"
#include "path_navigator.h"
#include "path_strategy.h"
#include "performance_monitor.h"
#include "shoot_controller.h"
#include "state_manager.h"
#include "time_manager.h"
#include "visualizer.h"
#include "waypoint_utils.h"
#include "yolov5_detector.h"

using waypoint_utils::healthTopicIndexToInternalIndex;
using waypoint_utils::kNumEnemies;

GoNode::~GoNode() = default;


/**
 * @brief 构造函数 —— 完成整个系统的初始化
 * @param player_id 玩家ID（1=蓝方，2=红方）
 * @param test_waypoint_idx 测试模式下的射击路径点编号（-1=正常模式，1-11=仅该点为射击点）
 *
 * 测试模式说明：
 *   正常模式下所有 has_shoot_task=true 的路径点都会执行射击。
 *   测试模式下只保留指定路径点的射击任务，其他路径点变为纯移动点。
 *   用于单点调试（如只测试 P6 的射击效果，传入 test_waypoint_idx=6）。
 */
GoNode::GoNode(int player_id, int test_waypoint_idx) : Node("go_node"), test_waypoint_idx_(test_waypoint_idx) {
    ctx_.player_id = player_id;

    // ── Step 1: 加载配置文件 ──
    std::string default_config_path = ament_index_cpp::get_package_share_directory("go") + "/config/game_config.yaml";
    this->declare_parameter<std::string>("config_path", default_config_path);
    this->declare_parameter<std::string>("engine_path", "/home/robot/competition/models/yolov5n_d2c2_fp16.engine");
    std::string config_path = this->get_parameter("config_path").as_string();
    std::string engine_path = this->get_parameter("engine_path").as_string();

    config_manager_ = std::make_unique<ConfigManager>();
    if (config_manager_->loadFromFile(config_path)) {
        RCLCPP_INFO(this->get_logger(), "配置文件加载成功");
    } else {
        throw std::runtime_error("配置文件加载失败：路径点没有代码后备值，无法安全启动");
    }

    loadConfigParameters();
    if (!validateConfig()) {
        throw std::runtime_error("配置验证失败，拒绝使用无效参数启动");
    }

    // ── Step 2: 从 YAML 创建路径策略 ──
    const auto& game_config = config_manager_->getConfig();
    const auto& configured_waypoints = player_id == 1 ? game_config.player1_waypoints : game_config.player2_waypoints;
    const size_t expected_waypoint_count = waypoint_utils::kBaseWaypointEnd + 1;
    if (configured_waypoints.size() != expected_waypoint_count) {
        throw std::runtime_error("当前导航和敌方映射要求每个阵营恰好配置 11 个路径点");
    }
    waypoints_ = configured_waypoints;  // 测试模式需要可修改副本
    strategy_ = PathStrategyFactory::createStrategy(waypoints_.size());

    if (test_waypoint_idx_ >= 0) {
        for (auto& wp : waypoints_) wp.has_shoot_task = false;
        size_t idx = static_cast<size_t>(test_waypoint_idx_) - 1;
        if (idx < waypoints_.size()) {
            waypoints_[idx].has_shoot_task = true;
            RCLCPP_INFO(this->get_logger(), "测试模式：仅P%d为射击点位", test_waypoint_idx_);
        }
    }

    // ── Step 3: 创建基础模块 ──
    // 注意：这些模块之间有依赖关系，顺序不能乱
    const ShootControllerConfig shoot_config{
        game_config.swing.range,
        game_config.swing.speed,
        game_config.threshold.consecutive_miss_disable,
        game_config.threshold.consecutive_miss_swing,
    };
    const YOLOv5DetectorConfig detector_config{
        config_.detection_conf_threshold,
        config_.detection_nms_iou,
        game_config.camera.fx,
        game_config.camera.fy,
        game_config.camera.cx,
        game_config.camera.cy,
        game_config.ballistics.bullet_speed,
        game_config.ballistics.gravity,
        game_config.armor.width,
        game_config.armor.height,
        game_config.filter.distance_window,
    };
    const auto threshold_or = [&game_config](const char* key, int fallback) {
        const auto it = game_config.health_change_thresholds.find(key);
        return it == game_config.health_change_thresholds.end() ? fallback : it->second;
    };
    const waypoint_utils::HealthChangeThresholdConfig health_thresholds{
        threshold_or("p4", waypoint_utils::HEALTH_CHANGE_THRESHOLD_P4),
        threshold_or("p5", waypoint_utils::HEALTH_CHANGE_THRESHOLD_P5),
        threshold_or("p6", waypoint_utils::HEALTH_CHANGE_THRESHOLD_P6),
        threshold_or("base", waypoint_utils::HEALTH_CHANGE_THRESHOLD_BASE),
        game_config.waypoint.health_change_fallback,
    };

    data_manager_ = std::make_unique<DataManager>(config_.health_history_duration, config_.other_data_duration,
                                                  config_.data_max_health_change, config_.data_health_tolerance,
                                                  config_.data_max_health_changes, config_.data_yaw_lookback);
    path_navigator_ =
        std::make_unique<PathNavigator>(*strategy_, game_config.timeout.move, config_.navigator_max_retry);
    shoot_controller_ = std::make_unique<ShootController>(shoot_config);
    yolov5_detector_ = std::make_unique<YOLOv5Detector>(player_id, engine_path, detector_config);
    performance_monitor_ = std::make_unique<PerformanceMonitor>(100, this->get_logger());

    // 可视化模块仅在 YAML 启用时创建；无头服务器应将该配置关闭。
    if (config_.enable_visualization) {
        visualizer_ = std::make_unique<Visualizer>(std::chrono::milliseconds(config_.timer_visualization));
    }

    // ── Step 4: 创建回调组（线程隔离） ──
    // control_cb_group_: 仅 5ms 控制定时器使用，不被其他回调阻塞
    // other_cb_group_:   其余 4 个定时器 + 5 个订阅共用
    control_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    other_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // ── Step 5: 订阅模拟器话题（全部归入 other_cb_group_） ──
    // 根据 player_id 选择对应玩家的话题（双方共享 /game_healths 和 /game_time）
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = other_cb_group_;

    std::string position_topic = (player_id == 1) ? "/position_player_1" : "/position_player_2";
    std::string angles_topic = (player_id == 1) ? "/real_angles_player_1" : "/real_angles_player_2";

    position_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        position_topic, 10, std::bind(&GoNode::positionCallback, this, std::placeholders::_1), sub_opts);

    real_angles_sub_ = this->create_subscription<tdt_interface::msg::ReceiveData>(
        angles_topic, 10, std::bind(&GoNode::anglesCallback, this, std::placeholders::_1), sub_opts);

    health_sub_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
        "/game_healths", 10, std::bind(&GoNode::healthCallback, this, std::placeholders::_1), sub_opts);

    std::string camera_topic = (player_id == 1) ? "/camera_image_player_1" : "/camera_image_player_2";
    camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        camera_topic, 10, std::bind(&GoNode::cameraCallback, this, std::placeholders::_1), sub_opts);

    game_time_sub_ = this->create_subscription<std_msgs::msg::Int32>(
        "/game_time", 10, std::bind(&GoNode::gameTimeCallback, this, std::placeholders::_1), sub_opts);

    // ── Step 6: 初始化时间管理器 ──
    TimeManager::initialize(this->now());

    // ── Step 7: 创建发布者 ──
    std::string speed_topic = (player_id == 1) ? "/target_speed_player_1" : "/target_speed_player_2";
    std::string target_angles_topic = (player_id == 1) ? "/target_angles_player_1" : "/target_angles_player_2";

    speed_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(speed_topic, 10);
    angles_pub_ = this->create_publisher<tdt_interface::msg::SendData>(target_angles_topic, 10);

    // ── Step 8: 组装三大控制器 ──
    // MovementController：底盘运动控制（速度指令）
    movement_ =
        std::make_unique<MovementController>(ctx_, config_, speed_pub_, angles_pub_, *data_manager_, *path_navigator_,
                                             *shoot_controller_, waypoints_, this->get_logger());

    // CombatManager：自瞄检测+射击指令发布
    combat_ = std::make_unique<CombatManager>(ctx_, config_, angles_pub_, *yolov5_detector_, *performance_monitor_,
                                              *path_navigator_, *shoot_controller_, waypoints_, this->get_logger());

    // StateManager：血量监控+自瞄降级+被攻击响应
    state_mgr_ = std::make_unique<StateManager>(ctx_, *shoot_controller_, *data_manager_, *path_navigator_,
                                                performance_monitor_.get(), health_thresholds, config_, waypoints_,
                                                angles_pub_, this->get_logger());

    // ── Step 9: 启动定时器 ──
    // 5个定时器分别驱动不同子系统的周期性循环
    // control_timer_ 分配独立回调组，确保 5ms 周期不会被其他回调阻塞
    control_timer_ = this->create_wall_timer(std::chrono::milliseconds(config_.timer_control),
                                             std::bind(&GoNode::controlLoop, this), control_cb_group_);

    shoot_timer_ = this->create_wall_timer(std::chrono::milliseconds(config_.timer_shoot),
                                           std::bind(&GoNode::shootLoop, this), other_cb_group_);

    position_adjust_timer_ =
        this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(config_.position_adjust_interval * 1000)),
                                std::bind(&GoNode::positionAdjustLoop, this), other_cb_group_);

    health_check_timer_ = this->create_wall_timer(std::chrono::milliseconds(config_.timer_health_check),
                                                  std::bind(&GoNode::healthCheckLoop, this), other_cb_group_);

    swing_timer_ = this->create_wall_timer(std::chrono::milliseconds(config_.timer_swing),
                                           std::bind(&GoNode::swingLoop, this), other_cb_group_);

    RCLCPP_INFO(this->get_logger(), "GoNode initialized for Player %d", player_id);
}


/**
 * @brief 将 YAML 配置值加载到 RuntimeConfig 结构体
 *
 * 注意：部分参数需要乘以 meter_to_map_unit 单位换算系数
 * （YAML 中以"米"为单位配置，运行时需要转换为"地图单位"）
 */
void GoNode::loadConfigParameters() {
    if (config_manager_ && config_manager_->isLoaded()) {
        const auto& config = config_manager_->getConfig();

        // 速度参数：需要单位换算
        config_.max_speed_far = config.speed.max_far * config.unit.meter_to_map_unit;
        config_.max_speed_near = config.speed.max_near * config.unit.meter_to_map_unit;
        config_.precise_adjust_speed = config.speed.precise_adjust_speed * config.unit.meter_to_map_unit;

        // 距离参数：直接使用（已经是地图单位）
        config_.speed_switch_distance = config.speed.switch_distance;
        config_.speed_switch_distance_fast = config.speed.switch_distance_fast;
        config_.precise_mode_distance = config.speed.precise_mode_distance;
        config_.arrival_threshold_fast = config.speed.arrival_threshold_fast;
        config_.arrival_threshold_normal = config.speed.arrival_threshold_normal;
        config_.precise_arrival_threshold = config.speed.precise_arrival_threshold;

        // 位置保持参数
        config_.position_hold_threshold = config.position_adjust.hold_threshold;
        config_.position_adjust_interval = config.position_adjust.adjust_interval;

        // 超时和阈值参数
        config_.health_history_duration = config.timeout.health_history;
        config_.other_data_duration = config.timeout.other_data;
        config_.detection_fail_max = config.detection_failure.max_fail_count;

        // 距离门控参数（无需单位换算，直接在代码中使用）
        config_.default_detection_distance = config.distance_gate.default_detection_distance;
        config_.distance_gate_max = config.distance_gate.gate_max;
        config_.distance_gate_acceptance = config.distance_gate.gate_acceptance;

        // ── EKF 参数 ──
        config_.ekf_q_yaw = config.ekf.q_yaw;
        config_.ekf_q_pitch = config.ekf.q_pitch;
        config_.ekf_q_distance = config.ekf.q_distance;
        config_.ekf_r_yaw = config.ekf.r_yaw;
        config_.ekf_r_pitch = config.ekf.r_pitch;
        config_.ekf_r_distance_denom = config.ekf.r_distance_denom;
        config_.ekf_r_distance_base = config.ekf.r_distance_base;
        config_.ekf_p0_yaw = config.ekf.p0_yaw;
        config_.ekf_p0_pitch = config.ekf.p0_pitch;
        config_.ekf_p0_distance = config.ekf.p0_distance;
        config_.ekf_p_reset = config.ekf.p_reset;
        config_.ekf_q_adaptive_alpha = config.ekf.q_adaptive_alpha;
        config_.ekf_q_adaptive_max_scale = config.ekf.q_adaptive_max_scale;

        // ── NIS 参数 ──
        config_.nis_chi2_upper = config.nis.chi2_upper;
        config_.nis_window = config.nis.window;

        // ── 跟踪状态机参数 ──
        config_.tracking_min_detect = config.tracking.min_detect_count;
        config_.tracking_max_temp_lost = config.tracking.max_temp_lost_count;
        config_.tracking_nis_failure_max = config.tracking.nis_failure_max;

        // ── 检测参数 ──
        config_.detection_conf_threshold = config.detection.conf_threshold;
        config_.detection_nms_iou = config.detection.nms_iou_threshold;
        config_.enable_visualization = config.detection.enable_visualization;

        // ── 定时器间隔（毫秒） ──
        config_.timer_control = config.timer.control;
        config_.timer_shoot = config.timer.shoot;
        config_.timer_health_check = config.timer.health_check;
        config_.timer_swing = config.timer.swing;
        config_.timer_visualization = config.timer.visualization;

        // ── 漂移检测参数 ──
        config_.drift_threshold_multiplier = config.drift.threshold_multiplier;
        config_.drift_timeout = config.drift.timeout;

        // ── 锁定加速倍率 ──
        config_.lock_speed_multiplier = config.lock.speed_multiplier;

        // ── 数据管理器参数 ──
        config_.data_max_health_change = config.data.max_reasonable_health_change;
        config_.data_health_tolerance = config.data.health_history_tolerance;
        config_.data_max_health_changes = config.data.max_health_changes;
        config_.data_yaw_lookback = config.data.yaw_pitch_lookback;

        // ── 导航器参数 ──
        config_.navigator_max_retry = config.navigator.max_retry_count;

        // ── 战斗管理器参数 ──
        config_.combat_ekf_default_dt = config.combat.ekf_default_dt;
        config_.combat_ekf_max_dt = config.combat.ekf_max_dt;
        config_.combat_debug_log_interval = config.combat.debug_log_interval;

        // ── 路点工具参数 ──
        config_.waypoint_health_ratio_escape = config.waypoint.health_ratio_escape;
        config_.waypoint_health_change_fallback = config.waypoint.health_change_fallback;
    }
}


/**
 * @brief 配置参数校验模板函数
 * @param value 参数值
 * @param min 最小值（开区间，value 必须 > min）
 * @param max 最大值（闭区间，value 必须 ≤ max）
 * @param name 参数名（用于错误日志）
 * @param include_min true 时最小值也合法，区间改为 [min, max]
 * @return true=合法，false=非法
 *
 * 校验规则是 (min, max] 左开右闭区间，
 * 即 value=0 的参数必须 >0 才能通过（防止除零等数学错误）
 */
template <typename T>
static bool validateField(T value, T min, T max, const char* name, const rclcpp::Logger& logger,
                          bool include_min = false) {
    bool invalid = include_min ? value < min : value <= min;
    invalid = invalid || value > max;
    if constexpr (std::is_floating_point_v<T>) {
        invalid = invalid || !std::isfinite(value);
    }

    if (invalid) {
        if constexpr (std::is_same_v<T, int>) {
            RCLCPP_ERROR(logger, "配置错误：%s = %d 超出范围 %c%d, %d]", name, value, include_min ? '[' : '(',
                         static_cast<int>(min), static_cast<int>(max));
        } else {
            RCLCPP_ERROR(logger, "配置错误：%s = %.4g 超出范围 %c%.4g, %.4g] 或不是有限数", name,
                         static_cast<double>(value), include_min ? '[' : '(', static_cast<double>(min),
                         static_cast<double>(max));
        }
        return false;
    }
    return true;
}


/**
 * @brief 批量校验所有关键配置参数
 *
 * 校验约束关系：
 *   max_speed_near ≤ max_speed_far
 *   precise_mode_distance ≤ speed_switch_distance
 *   arrival_threshold_normal ≤ arrival_threshold_fast
 *   precise_arrival_threshold ≤ arrival_threshold_normal
 *   position_hold_threshold ≤ precise_arrival_threshold
 *   precise_adjust_speed ≤ max_speed_near
 *
 * 这些约束保证了速度曲线的单调性和距离阈值的层次关系
 */
bool GoNode::validateConfig() {
    bool is_valid = true;
    auto logger = this->get_logger();
    const auto& raw = config_manager_->getConfig();

    is_valid &= validateField(config_.max_speed_far, 0.0, 100.0, "max_speed_far", logger);
    is_valid &= validateField(config_.max_speed_near, 0.0, config_.max_speed_far, "max_speed_near", logger);
    is_valid &= validateField(config_.speed_switch_distance, 0.0, 50.0, "speed_switch_distance", logger);
    is_valid &= validateField(config_.speed_switch_distance_fast, 0.0, 100.0, "speed_switch_distance_fast", logger);
    is_valid &= validateField(config_.precise_mode_distance, 0.0, config_.speed_switch_distance,
                              "precise_mode_distance", logger);
    is_valid &= validateField(config_.arrival_threshold_fast, 0.0, 20.0, "arrival_threshold_fast", logger);
    is_valid &= validateField(config_.arrival_threshold_normal, 0.0, config_.arrival_threshold_fast,
                              "arrival_threshold_normal", logger);
    is_valid &=
        validateField(config_.precise_adjust_speed, 0.0, config_.max_speed_near, "precise_adjust_speed", logger);
    is_valid &= validateField(config_.position_adjust_interval, 0.0, 60.0, "position_adjust_interval", logger);
    is_valid &= validateField(config_.health_history_duration, 0.0, 60.0, "health_history_duration", logger);
    is_valid &= validateField(config_.other_data_duration, 0.0, 60.0, "other_data_duration", logger);
    is_valid &= validateField(config_.detection_fail_max, 0, 1000, "detection_fail_max_", logger);
    is_valid &= validateField(config_.precise_arrival_threshold, 0.0, config_.arrival_threshold_normal,
                              "precise_arrival_threshold", logger);
    is_valid &= validateField(config_.position_hold_threshold, 0.0, config_.precise_arrival_threshold,
                              "position_hold_threshold", logger);
    is_valid &= validateField(config_.default_detection_distance, 0.0, 1000.0, "default_detection_distance", logger);
    is_valid &= validateField(config_.distance_gate_max, 0.0, 1000.0, "distance_gate_max", logger);
    is_valid &= validateField(config_.distance_gate_acceptance, 0.0, 1000.0, "distance_gate_acceptance", logger, true);
    if (config_.default_detection_distance > config_.distance_gate_max) {
        RCLCPP_ERROR(logger, "配置错误：default_detection_distance 不能大于 distance_gate_max");
        is_valid = false;
    }

    // EKF / NIS：R、P 和距离噪声分母必须为正，避免奇异协方差或除零。
    is_valid &= validateField(config_.ekf_q_yaw, 0.0, 1.0, "ekf_q_yaw", logger);
    is_valid &= validateField(config_.ekf_q_pitch, 0.0, 1.0, "ekf_q_pitch", logger);
    is_valid &= validateField(config_.ekf_q_distance, 0.0, 10.0, "ekf_q_distance", logger);
    is_valid &= validateField(config_.ekf_r_yaw, 0.0, 100.0, "ekf_r_yaw", logger);
    is_valid &= validateField(config_.ekf_r_pitch, 0.0, 100.0, "ekf_r_pitch", logger);
    is_valid &= validateField(config_.ekf_r_distance_denom, 0.0, 1e9, "ekf_r_distance_denom", logger);
    is_valid &= validateField(config_.ekf_r_distance_base, 0.0, 100.0, "ekf_r_distance_base", logger);
    is_valid &= validateField(config_.ekf_p0_yaw, 0.0, 1e9, "ekf_p0_yaw", logger);
    is_valid &= validateField(config_.ekf_p0_pitch, 0.0, 1e9, "ekf_p0_pitch", logger);
    is_valid &= validateField(config_.ekf_p0_distance, 0.0, 1e9, "ekf_p0_distance", logger);
    is_valid &= validateField(config_.ekf_p_reset, 0.0, 1e9, "ekf_p_reset", logger);
    is_valid &= validateField(config_.ekf_q_adaptive_alpha, 0.0, 100.0, "ekf_q_adaptive_alpha", logger, true);
    is_valid &= validateField(config_.ekf_q_adaptive_max_scale, 1.0, 1000.0, "ekf_q_adaptive_max_scale", logger, true);
    is_valid &= validateField(config_.nis_chi2_upper, 0.0, 20.0, "nis_chi2_upper", logger);
    is_valid &= validateField(config_.nis_window, 0, 1000, "nis_window", logger);

    // 状态机、检测和定时器。
    is_valid &= validateField(config_.tracking_min_detect, 0, 1000, "tracking_min_detect", logger);
    is_valid &= validateField(config_.tracking_max_temp_lost, 0, 1000, "tracking_max_temp_lost", logger);
    is_valid &= validateField(config_.tracking_nis_failure_max, 0.0, 1.0, "tracking_nis_failure_max", logger);
    is_valid &= validateField(config_.detection_conf_threshold, 0.0, 1.0, "detection_conf_threshold", logger);
    is_valid &= validateField(config_.detection_nms_iou, 0.0, 1.0, "detection_nms_iou", logger);
    is_valid &= validateField(config_.timer_control, 0, 1000, "timer_control", logger);
    is_valid &= validateField(config_.timer_shoot, 0, 1000, "timer_shoot", logger);
    is_valid &= validateField(config_.timer_health_check, 0, 10000, "timer_health_check", logger);
    is_valid &= validateField(config_.timer_swing, 0, 1000, "timer_swing", logger);
    is_valid &= validateField(config_.timer_visualization, 0, 1000, "timer_visualization", logger);

    // 其余运行参数。
    is_valid &= validateField(config_.drift_threshold_multiplier, 0.0, 100.0, "drift_threshold_multiplier", logger);
    is_valid &= validateField(config_.drift_timeout, 0.0, 3600.0, "drift_timeout", logger);
    is_valid &= validateField(config_.lock_speed_multiplier, 0.0, 100.0, "lock_speed_multiplier", logger);
    is_valid &= validateField(config_.data_max_health_change, 0, 100000, "data_max_health_change", logger);
    is_valid &= validateField(config_.data_health_tolerance, 0.0, 60.0, "data_health_tolerance", logger, true);
    is_valid &= validateField(config_.data_max_health_changes, 0, 10000, "data_max_health_changes", logger);
    is_valid &= validateField(config_.data_yaw_lookback, 0.0, 60.0, "data_yaw_lookback", logger);
    is_valid &= validateField(config_.navigator_max_retry, 0, 1000, "navigator_max_retry", logger);
    is_valid &= validateField(config_.combat_ekf_default_dt, 0.0, 10.0, "combat_ekf_default_dt", logger);
    is_valid &= validateField(config_.combat_ekf_max_dt, 0.0, 60.0, "combat_ekf_max_dt", logger);
    is_valid &= validateField(config_.combat_debug_log_interval, 0, 1000000, "combat_debug_log_interval", logger);
    is_valid &=
        validateField(config_.waypoint_health_ratio_escape, 0.0, 1.0, "waypoint_health_ratio_escape", logger, true);
    if (config_.combat_ekf_max_dt < config_.combat_ekf_default_dt) {
        RCLCPP_ERROR(logger, "配置错误：combat_ekf_max_dt 不能小于 combat_ekf_default_dt");
        is_valid = false;
    }

    // 仅保存在 GameConfig 中、由具体模块直接读取的参数。
    is_valid &= validateField(raw.unit.meter_to_map_unit, 0.0, 1000.0, "unit.meter_to_map_unit", logger);
    is_valid &= validateField(raw.camera.fx, 0.0, 100000.0, "camera.fx", logger);
    is_valid &= validateField(raw.camera.fy, 0.0, 100000.0, "camera.fy", logger);
    is_valid &= validateField(raw.camera.cx, 0.0, 100000.0, "camera.cx", logger, true);
    is_valid &= validateField(raw.camera.cy, 0.0, 100000.0, "camera.cy", logger, true);
    is_valid &= validateField(raw.armor.width, 0.0, 1000.0, "armor.width", logger);
    is_valid &= validateField(raw.armor.height, 0.0, 1000.0, "armor.height", logger);
    is_valid &= validateField(raw.ballistics.bullet_speed, 0.0, 10000.0, "ballistics.bullet_speed", logger);
    is_valid &= validateField(raw.ballistics.gravity, 0.0, 10000.0, "ballistics.gravity", logger, true);
    is_valid &= validateField(raw.swing.range, 0.0, 180.0, "swing.range", logger, true);
    is_valid &= validateField(raw.swing.speed, 0.0, 10000.0, "swing.speed", logger, true);
    is_valid &= validateField(raw.timeout.move, 0.0, 3600.0, "timeout.move", logger);
    is_valid &=
        validateField(raw.threshold.consecutive_miss_disable, 0, 100000, "threshold.consecutive_miss_disable", logger);
    is_valid &=
        validateField(raw.threshold.consecutive_miss_swing, 0, 100000, "threshold.consecutive_miss_swing", logger);
    is_valid &= validateField(raw.filter.distance_window, 0, 100000, "filter.distance_window", logger);

    if (is_valid) {
        RCLCPP_INFO(logger, "配置验证通过");
    } else {
        RCLCPP_ERROR(logger, "配置验证失败，请检查配置文件");
    }

    return is_valid;
}


// ══════════════════════════════════════════════════════════════
//  ROS2 回调函数 —— 接收模拟器数据，更新内部状态
// ══════════════════════════════════════════════════════════════

/**
 * @brief 位置回调 —— 更新机器人在世界坐标系中的 (x, y)
 * 由 /position_player_X 话题触发
 */
void GoNode::positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if (!msg) {
        RCLCPP_ERROR(this->get_logger(), "位置消息为空");
        return;
    }
    data_manager_->updatePosition(msg->pose.position.x, msg->pose.position.y, msg->header.stamp);
}


/**
 * @brief 云台角度回调 —— 更新当前云台 yaw/pitch
 * 由 /real_angles_player_X 话题触发
 * 同时更新 SharedContext（供其他线程读取）和 DataManager（供历史记录查询）
 */
void GoNode::anglesCallback(const tdt_interface::msg::ReceiveData::SharedPtr msg) {
    if (!msg) {
        RCLCPP_ERROR(this->get_logger(), "角度消息为空");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(ctx_.gimbal.mutex);
        ctx_.gimbal.value.current_yaw = msg->yaw;
        ctx_.gimbal.value.current_pitch = msg->pitch;
    }
    data_manager_->updateYawPitch(msg->yaw, msg->pitch, msg->timestamp);
}


/**
 * @brief 血量回调 —— 更新自身和所有敌人的血量
 * 由 /game_healths 话题触发（12个int32的数组）
 *
 * 血量数组索引映射（固定，与player_id无关）：
 *   [0]  蓝方玩家    [1]  蓝方3号    [2]  蓝方4号    [3]  蓝方5号
 *   [4]  蓝方前哨站  [5]  蓝方基地
 *   [6]  红方玩家    [7]  红方3号    [8]  红方4号    [9]  红方5号
 *   [10] 红方前哨站  [11] 红方基地
 *
 * player_id=1(蓝方)：
 *   自身 → index 0, 敌人 → index 6-11（映射为内部索引 0-5）
 * player_id=2(红方)：
 *   自身 → index 6, 敌人 → index 0-5（直接使用）
 *
 * 每次收到血量更新还会检查基地是否被摧毁
 */
void GoNode::healthCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg) {
    if (!msg) {
        RCLCPP_ERROR(this->get_logger(), "血量消息为空");
        return;
    }
    auto current_time = TimeManager::getCurrentTime();

    // 更新自身血量
    size_t self_index = (ctx_.player_id == 1) ? 0 : 6;
    if (self_index < msg->data.size()) {
        data_manager_->updateSelfHealth(msg->data[self_index], current_time);
    }

    // 更新6个敌方单位的血量
    for (int i = 0; i < kNumEnemies; i++) {
        size_t enemy_index = (ctx_.player_id == 1) ? (6 + i) : i;
        if (enemy_index < msg->data.size()) {
            data_manager_->updateEnemyHealth(i, msg->data[enemy_index], current_time);
            checkBaseDestroyed(i, msg->data[enemy_index]);
        }
    }
}

/**
 * @brief 基地摧毁检测 —— 我方基地血量归零时终止程序
 * @param enemy_internal_index 当前更新的敌人在内部数组中的索引
 * @param enemy_health 该敌人的当前血量
 *
 * 判断条件：
 *   - 更新的是"我方基地"对应的内部索引
 *   - 基地血量 ≤ 0
 *   - 基地血量已初始化（排除初始值0的误判）
 *
 * 注意：检查的是我方基地被摧毁（游戏失败），不是敌方基地
 */
void GoNode::checkBaseDestroyed(size_t enemy_internal_index, int enemy_health) {
    size_t base_topic_index = (ctx_.player_id == 1) ? 11 : 5;
    int base_internal_index = healthTopicIndexToInternalIndex(base_topic_index, ctx_.player_id);

    bool index_valid = base_internal_index >= 0 && base_internal_index < kNumEnemies;
    bool is_base_target = enemy_internal_index == static_cast<size_t>(base_internal_index);
    bool is_destroyed = enemy_health <= 0;
    bool health_initialized = data_manager_->isEnemyHealthInitialized(base_internal_index);

    if (index_valid && is_base_target && is_destroyed && health_initialized) {
        RCLCPP_FATAL(this->get_logger(), "基地被摧毁！游戏结束！");
        game_over_ = true;
    }
}


/**
 * @brief 相机图像回调 —— 将 ROS Image 消息转换为 OpenCV Mat 并更新检测器
 * 由 /camera_image_player_X 话题触发
 *
 * 使用 cv_bridge 将 sensor_msgs/Image 转为 BGR8 格式的 OpenCV Mat
 * 转换后的图像存入 YOLOv5Detector 的 latest_frame_，
 * 供 shootLoop 中的推理线程异步处理
 *
 * 异常处理：cv_bridge 转换失败、图像为空、尺寸无效等情况均只打日志不崩溃
 */
void GoNode::cameraCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
        if (!msg) {
            RCLCPP_ERROR(this->get_logger(), "相机消息为空");
            return;
        }
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        if (cv_ptr->image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "图像为空");
            return;
        }
        if (cv_ptr->image.cols <= 0 || cv_ptr->image.rows <= 0) {
            RCLCPP_ERROR(this->get_logger(), "图像尺寸无效: %dx%d", cv_ptr->image.cols, cv_ptr->image.rows);
            return;
        }
        yolov5_detector_->updateFrame(cv_ptr->image);
        if (visualizer_) {
            visualizer_->update(cv_ptr->image, yolov5_detector_->getCachedArmors());
        }
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "相机回调异常: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "相机回调未知异常");
    }
}

/**
 * @brief 游戏时间回调 —— 更新全局游戏时钟
 * 由 /game_time 话题触发（每秒一次，从0开始递增）
 */
void GoNode::gameTimeCallback(const std_msgs::msg::Int32::SharedPtr msg) {
    if (!msg) {
        RCLCPP_ERROR(this->get_logger(), "游戏时间消息为空");
        return;
    }
    TimeManager::updateGameTime(msg->data);
}


// ══════════════════════════════════════════════════════════════
//  定时器回调 —— 驱动各子系统的周期性循环
// ══════════════════════════════════════════════════════════════

/**
 * @brief 运动控制定时器回调（5ms / 200Hz）
 * 检测到基地被摧毁时调用 rclcpp::shutdown() 终止程序
 */
void GoNode::controlLoop() {
    if (game_over_) {
        rclcpp::shutdown();
        return;
    }
    movement_->controlLoop();
}

/** @brief 自瞄检测定时器回调（30ms / ~33Hz）*/
void GoNode::shootLoop() { combat_->shootLoop(); }

/** @brief 位置保持定时器回调（可配置间隔）*/
void GoNode::positionAdjustLoop() { movement_->positionAdjustLoop(); }

/** @brief 血量监控定时器回调（1s / 1Hz）*/
void GoNode::healthCheckLoop() { state_mgr_->healthCheckLoop(); }

/** @brief 摆动+射击指令定时器回调（20ms / 50Hz）*/
void GoNode::swingLoop() { combat_->swingLoop(); }
