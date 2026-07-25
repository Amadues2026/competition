/**
 * @file config_manager.cpp
 * @brief YAML 配置文件解析器 —— 将 game_config.yaml 加载为强类型 GameConfig 结构体
 *
 * ConfigManager 负责把用户可编辑的 YAML 配置转化为代码可直接使用的结构体。
 * 所有模块通过 config_manager.getConfig() 获取配置值，避免硬编码常量分散各处。
 *
 * 支持的配置分类（对应 game_config.yaml 中的各顶层 key）：
 *   - speed:              运动控制参数（最大速度、距离阈值等，单位为米）
 *   - camera:             相机内参（fx, fy, cx, cy，用于 PnP 求解）
 *   - armor:              装甲板尺寸（宽/高，米，用于 PnP 求解）
 *   - ballistics:         弹道参数（弹丸初速、重力加速度）
 *   - position_adjust:    射击位置保持参数
 *   - swing:              摆动扫描参数（角度范围、速度）
 *   - detection_failure:  检测失败容忍参数
 *   - timeout:            各类超时时间（移动超时、血量历史时长等）
 *   - threshold:          连续未命中阈值（用于禁用/摆动模式切换）
 *   - filter:             滤波器窗口大小
 *   - unit:               单位换算系数（米→地图单位）
 *   - ekf:                EKF 滤波器参数（过程/测量噪声、协方差等）
 *   - nis:                NIS 卡方检验参数
 *   - tracking:           跟踪状态机参数
 *   - detection:           YOLO 置信度/NMS IoU 阈值
 *   - timer:              各定时器间隔（毫秒）
 *   - drift:              射击漂移检测参数
 *   - lock:               目标锁定加速倍率
 *   - data:               数据管理器常量
 *   - navigator:          导航器卡住重试上限
 *   - combat:             EKF dt 默认值和调试日志间隔
 *   - waypoint:           路点工具常量
 *   - player1_waypoints/player2_waypoints: 两个阵营的路径点序列
 *   - health_change_thresholds: 血量变化阈值（各路径点对应的不同阈值）
 *
 * 错误处理策略：
 *   - YAML 解析异常（文件不存在/格式错误）→ 打日志返回 false，由调用方回退到代码默认值
 *   - 单个字段缺失 → 抛 runtime_error，外层 catch 统一处理
 *   - 不做字段级容错：缺失即失败，强制用户检查配置文件
 */

#include "config_manager.h"

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <stdexcept>

namespace {

std::vector<Waypoint> parseWaypoints(const YAML::Node& node, const char* name);
std::unordered_map<std::string, int> parseHealthThresholds(const YAML::Node& node);
GameConfig::SpeedConfig parseSpeedConfig(const YAML::Node& node);
GameConfig::CameraConfig parseCameraConfig(const YAML::Node& node);
GameConfig::ArmorConfig parseArmorConfig(const YAML::Node& node);
GameConfig::BallisticsConfig parseBallisticsConfig(const YAML::Node& node);
GameConfig::PositionAdjustConfig parsePositionAdjustConfig(const YAML::Node& node);
GameConfig::SwingConfig parseSwingConfig(const YAML::Node& node);
GameConfig::DetectionFailureConfig parseDetectionFailureConfig(const YAML::Node& node);
GameConfig::TimeoutConfig parseTimeoutConfig(const YAML::Node& node);
GameConfig::ThresholdConfig parseThresholdConfig(const YAML::Node& node);
GameConfig::DistanceGateConfig parseDistanceGateConfig(const YAML::Node& node);
GameConfig::FilterConfig parseFilterConfig(const YAML::Node& node);
GameConfig::UnitConfig parseUnitConfig(const YAML::Node& node);
GameConfig::EkfConfig parseEkfConfig(const YAML::Node& node);
GameConfig::NisConfig parseNisConfig(const YAML::Node& node);
GameConfig::TrackingConfig parseTrackingConfig(const YAML::Node& node);
GameConfig::DetectionConfig parseDetectionConfig(const YAML::Node& node);
GameConfig::TimerConfig parseTimerConfig(const YAML::Node& node);
GameConfig::DriftConfig parseDriftConfig(const YAML::Node& node);
GameConfig::LockConfig parseLockConfig(const YAML::Node& node);
GameConfig::DataConfig parseDataConfig(const YAML::Node& node);
GameConfig::NavigatorConfig parseNavigatorConfig(const YAML::Node& node);
GameConfig::CombatConfig parseCombatConfig(const YAML::Node& node);
GameConfig::WaypointConfig parseWaypointConfig(const YAML::Node& node);

}  // namespace

/**
 * @brief 从指定路径加载 YAML 配置文件
 * @param config_path 配置文件绝对路径
 * @return true=加载成功，false=YAML 解析或读取失败
 *
 * 加载流程：
 *   1. YAML::LoadFile 解析文件为 Node 树
 *   2. 逐个解析各配置分类（调用本文件内的 parseXxx 函数）
 *   3. 设置 loaded_ = true，后续模块可通过 isLoaded() 判断配置是否可用
 *
 * 注意：parseXxx 函数对缺失字段会抛异常，此处统一 catch 后返回 false
 */
bool ConfigManager::loadFromFile(const std::string& config_path) {
    loaded_ = false;
    try {
        YAML::Node config = YAML::LoadFile(config_path);

        config_.player_id = config["player"]["id"].as<int>();
        config_.player1_waypoints = parseWaypoints(config["player1_waypoints"], "player1_waypoints");
        config_.player2_waypoints = parseWaypoints(config["player2_waypoints"], "player2_waypoints");

        config_.health_change_thresholds = parseHealthThresholds(config["health_change_thresholds"]);
        config_.speed = parseSpeedConfig(config["speed"]);
        config_.camera = parseCameraConfig(config["camera"]);
        config_.armor = parseArmorConfig(config["armor"]);
        config_.ballistics = parseBallisticsConfig(config["ballistics"]);
        config_.position_adjust = parsePositionAdjustConfig(config["position_adjust"]);
        config_.swing = parseSwingConfig(config["swing"]);
        config_.detection_failure = parseDetectionFailureConfig(config["detection_failure"]);
        config_.timeout = parseTimeoutConfig(config["timeout"]);
        config_.threshold = parseThresholdConfig(config["threshold"]);
        config_.distance_gate = parseDistanceGateConfig(config["distance_gate"]);
        config_.filter = parseFilterConfig(config["filter"]);
        config_.unit = parseUnitConfig(config["unit"]);

        config_.ekf = parseEkfConfig(config["ekf"]);
        config_.nis = parseNisConfig(config["nis"]);
        config_.tracking = parseTrackingConfig(config["tracking"]);
        config_.detection = parseDetectionConfig(config["detection"]);
        config_.timer = parseTimerConfig(config["timer"]);
        config_.drift = parseDriftConfig(config["drift"]);
        config_.lock = parseLockConfig(config["lock"]);
        config_.data = parseDataConfig(config["data"]);
        config_.navigator = parseNavigatorConfig(config["navigator"]);
        config_.combat = parseCombatConfig(config["combat"]);
        config_.waypoint = parseWaypointConfig(config["waypoint"]);
        loaded_ = true;
        std::cout << "配置文件加载成功: " << config_path << std::endl;
        return true;

    } catch (const YAML::Exception& e) {
        std::cerr << "YAML解析错误: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "配置加载错误: " << e.what() << std::endl;
        return false;
    }
}

namespace {

std::vector<Waypoint> parseWaypoints(const YAML::Node& node, const char* name) {
    if (!node.IsSequence() || node.size() == 0) {
        throw std::runtime_error(std::string(name) + " 必须是非空路径点序列");
    }

    std::vector<Waypoint> waypoints;
    waypoints.reserve(node.size());
    for (const auto& item : node) {
        if (!item.IsMap()) {
            throw std::runtime_error(std::string(name) + " 中的路径点必须是映射");
        }
        waypoints.push_back({item["x"].as<double>(), item["y"].as<double>(), item["yaw"].as<double>(),
                             item["pitch"].as<double>(), item["has_shoot_task"].as<bool>()});
    }
    return waypoints;
}

/**
 * @brief 解析血量变化阈值映射表
 * @return 以路径点名称为 key 的阈值 map（如 "p4" → 100, "base" → 50）
 *
 * YAML 格式示例：
 *   health_change_thresholds:
 *     p4: 100
 *     p5: 80
 *     p6: 60
 *     base: 50
 *
 * 这些阈值用于 StateManager 的自瞄降级决策：
 *   敌方血量变化量 > 阈值 → 说明敌人被我方其他武器打中 → 优先打它
 */
std::unordered_map<std::string, int> parseHealthThresholds(const YAML::Node& node) {
    std::unordered_map<std::string, int> thresholds;
    if (!node.IsMap()) {
        throw std::runtime_error("血量阈值配置必须是映射");
    }
    for (const auto& kv : node) {
        std::string key = kv.first.as<std::string>();
        int value = kv.second.as<int>();
        thresholds[key] = value;
    }
    return thresholds;
}

/**
 * @brief 解析运动控制速度配置
 * @return SpeedConfig 结构体，包含远/近距离最大速度、各距离阈值、精调速度
 *
 * 参数含义：
 *   max_far / max_near:          远/近距离下的底盘最大速度
 *   switch_distance:             远→近距离切换阈值
 *   switch_distance_fast:        快速移动模式下的切换阈值（更高）
 *   precise_mode_distance:       精调模式距离阈值（进入后降速提高精度）
 *   arrival_threshold_fast/normal/precise_arrival_threshold: 三层到达判定阈值
 *   precise_adjust_speed:        精调模式下的移动速度
 */
GameConfig::SpeedConfig parseSpeedConfig(const YAML::Node& node) {
    GameConfig::SpeedConfig speed;
    speed.max_far = node["max_far"].as<double>();
    speed.max_near = node["max_near"].as<double>();
    speed.switch_distance = node["switch_distance"].as<double>();
    speed.switch_distance_fast = node["switch_distance_fast"].as<double>();
    speed.precise_mode_distance = node["precise_mode_distance"].as<double>();
    speed.arrival_threshold_fast = node["arrival_threshold_fast"].as<double>();
    speed.arrival_threshold_normal = node["arrival_threshold_normal"].as<double>();
    speed.precise_arrival_threshold = node["precise_arrival_threshold"].as<double>();
    speed.precise_adjust_speed = node["precise_adjust_speed"].as<double>();
    return speed;
}

/**
 * @brief 解析相机内参配置
 * @return CameraConfig (fx, fy, cx, cy)，用于 PnP 单目测距
 */
GameConfig::CameraConfig parseCameraConfig(const YAML::Node& node) {
    GameConfig::CameraConfig camera;
    camera.fx = node["fx"].as<double>();
    camera.fy = node["fy"].as<double>();
    camera.cx = node["cx"].as<double>();
    camera.cy = node["cy"].as<double>();
    return camera;
}

/**
 * @brief 解析装甲板尺寸配置（米）
 * @return ArmorConfig (width, height)，用于 PnP 3D-2D 点对匹配
 */
GameConfig::ArmorConfig parseArmorConfig(const YAML::Node& node) {
    GameConfig::ArmorConfig armor;
    armor.width = node["width"].as<double>();
    armor.height = node["height"].as<double>();
    return armor;
}

/**
 * @brief 解析弹道参数配置
 * @return BallisticsConfig (bullet_speed, gravity)
 *   bullet_speed: 弹丸初速（m/s），用于 PnP 后的弹道补偿计算
 *   gravity:      重力加速度（m/s²），用于弹道下坠补偿
 */
GameConfig::BallisticsConfig parseBallisticsConfig(const YAML::Node& node) {
    GameConfig::BallisticsConfig ballistics;
    ballistics.bullet_speed = node["bullet_speed"].as<double>();
    ballistics.gravity = node["gravity"].as<double>();
    return ballistics;
}

/**
 * @brief 解析射击位置保持配置
 * @return PositionAdjustConfig
 *   hold_threshold:  到达射击点后的位移判定阈值（低于此值视为"已稳定"）
 *   adjust_interval: 射击期间位置修正的调用间隔（秒）
 */
GameConfig::PositionAdjustConfig parsePositionAdjustConfig(const YAML::Node& node) {
    GameConfig::PositionAdjustConfig pa;
    pa.hold_threshold = node["hold_threshold"].as<double>();
    pa.adjust_interval = node["adjust_interval"].as<double>();
    return pa;
}

/**
 * @brief 解析摆动扫描配置
 * @return SwingConfig
 *   range: 摆动角度范围（度，中心两侧各 ±range）
 *   speed: 摆动角速度（度/秒）
 */
GameConfig::SwingConfig parseSwingConfig(const YAML::Node& node) {
    GameConfig::SwingConfig swing;
    swing.range = node["range"].as<double>();
    swing.speed = node["speed"].as<double>();
    return swing;
}

/**
 * @brief 解析检测失败容忍配置
 * @return DetectionFailureConfig
 *   max_fail_count: 连续检测失败次数上限，超过则切换到摆动模式
 */
GameConfig::DetectionFailureConfig parseDetectionFailureConfig(const YAML::Node& node) {
    GameConfig::DetectionFailureConfig df;
    df.max_fail_count = node["max_fail_count"].as<int>();
    return df;
}

/**
 * @brief 解析超时配置
 * @return TimeoutConfig
 *   move:          移动到路径点的超时时间（秒），超过则跳过当前点
 *   health_history: 血量历史保留时长（秒），用于计算血量变化量
 *   other_data:    位置/角度历史保留时长（秒）
 */
GameConfig::TimeoutConfig parseTimeoutConfig(const YAML::Node& node) {
    GameConfig::TimeoutConfig timeout;
    timeout.move = node["move"].as<double>();
    timeout.health_history = node["health_history"].as<double>();
    timeout.other_data = node["other_data"].as<double>();
    return timeout;
}

/**
 * @brief 解析命中判定阈值配置
 * @return ThresholdConfig
 *   consecutive_miss_disable: 连续未命中次数超过此值 → 禁用自瞄（可能目标已消失）
 *   consecutive_miss_swing:   连续未命中次数超过此值 → 切换为摆动扫描模式
 */
GameConfig::ThresholdConfig parseThresholdConfig(const YAML::Node& node) {
    GameConfig::ThresholdConfig threshold;
    threshold.consecutive_miss_disable = node["consecutive_miss_disable"].as<int>();
    threshold.consecutive_miss_swing = node["consecutive_miss_swing"].as<int>();
    return threshold;
}

/**
 * @brief 解析距离门控参数配置
 * @return DistanceGateConfig
 *   default_detection_distance: 默认检测距离（无有效检测时用作回退）
 *   gate_max:                   距离门控上限，超过此距离的测量直接丢弃
 *   gate_acceptance:            帧间距离跳变容忍度，超过此值视为无效跳变
 */
GameConfig::DistanceGateConfig parseDistanceGateConfig(const YAML::Node& node) {
    GameConfig::DistanceGateConfig dg;
    dg.default_detection_distance = node["default_detection_distance"].as<double>();
    dg.gate_max = node["gate_max"].as<double>();
    dg.gate_acceptance = node["gate_acceptance"].as<double>();
    return dg;
}

/**
 * @brief 解析滤波器配置
 * @return FilterConfig
 *   distance_window: 距离中值滤波器的窗口大小（帧数）
 */
GameConfig::FilterConfig parseFilterConfig(const YAML::Node& node) {
    GameConfig::FilterConfig filter;
    filter.distance_window = node["distance_window"].as<int>();
    return filter;
}

/**
 * @brief 解析单位换算配置
 * @return UnitConfig
 *   meter_to_map_unit: 1米对应多少地图坐标单位（如 5.0 表示1米=5个地图单位）
 *
 * YAML 中距离参数以米为单位配置（人类可读），
 * 运行时需要乘以此系数转换为地图坐标单位（与模拟器对齐）
 */
GameConfig::UnitConfig parseUnitConfig(const YAML::Node& node) {
    GameConfig::UnitConfig unit;
    unit.meter_to_map_unit = node["meter_to_map_unit"].as<double>();
    return unit;
}

GameConfig::EkfConfig parseEkfConfig(const YAML::Node& node) {
    GameConfig::EkfConfig ekf;
    ekf.q_yaw = node["q_yaw"].as<double>();
    ekf.q_pitch = node["q_pitch"].as<double>();
    ekf.q_distance = node["q_distance"].as<double>();
    ekf.r_yaw = node["r_yaw"].as<double>();
    ekf.r_pitch = node["r_pitch"].as<double>();
    ekf.r_distance_denom = node["r_distance_denom"].as<double>();
    ekf.r_distance_base = node["r_distance_base"].as<double>();
    ekf.p0_yaw = node["p0_yaw"].as<double>();
    ekf.p0_pitch = node["p0_pitch"].as<double>();
    ekf.p0_distance = node["p0_distance"].as<double>();
    ekf.p_reset = node["p_reset"].as<double>();
    ekf.q_adaptive_alpha = node["q_adaptive_alpha"].as<double>();
    ekf.q_adaptive_max_scale = node["q_adaptive_max_scale"].as<double>();
    return ekf;
}

GameConfig::NisConfig parseNisConfig(const YAML::Node& node) {
    GameConfig::NisConfig nis;
    nis.chi2_upper = node["chi2_upper"].as<double>();
    nis.window = node["window"].as<int>();
    return nis;
}

GameConfig::TrackingConfig parseTrackingConfig(const YAML::Node& node) {
    GameConfig::TrackingConfig tracking;
    tracking.min_detect_count = node["min_detect_count"].as<int>();
    tracking.max_temp_lost_count = node["max_temp_lost_count"].as<int>();
    tracking.nis_failure_max = node["nis_failure_max"].as<double>();
    return tracking;
}

GameConfig::DetectionConfig parseDetectionConfig(const YAML::Node& node) {
    GameConfig::DetectionConfig detection;
    detection.conf_threshold = node["conf_threshold"].as<double>();
    detection.nms_iou_threshold = node["nms_iou_threshold"].as<double>();
    detection.enable_visualization = node["enable_visualization"].as<bool>();
    return detection;
}

GameConfig::TimerConfig parseTimerConfig(const YAML::Node& node) {
    GameConfig::TimerConfig timer;
    timer.control = node["control"].as<int>();
    timer.shoot = node["shoot"].as<int>();
    timer.health_check = node["health_check"].as<int>();
    timer.swing = node["swing"].as<int>();
    timer.visualization = node["visualization"].as<int>();
    return timer;
}

GameConfig::DriftConfig parseDriftConfig(const YAML::Node& node) {
    GameConfig::DriftConfig drift;
    drift.threshold_multiplier = node["threshold_multiplier"].as<double>();
    drift.timeout = node["timeout"].as<double>();
    return drift;
}

GameConfig::LockConfig parseLockConfig(const YAML::Node& node) {
    GameConfig::LockConfig lock;
    lock.speed_multiplier = node["speed_multiplier"].as<double>();
    return lock;
}

GameConfig::DataConfig parseDataConfig(const YAML::Node& node) {
    GameConfig::DataConfig data;
    data.max_reasonable_health_change = node["max_reasonable_health_change"].as<int>();
    data.health_history_tolerance = node["health_history_tolerance"].as<double>();
    data.max_health_changes = node["max_health_changes"].as<int>();
    data.yaw_pitch_lookback = node["yaw_pitch_lookback"].as<double>();
    return data;
}

GameConfig::NavigatorConfig parseNavigatorConfig(const YAML::Node& node) {
    GameConfig::NavigatorConfig navigator;
    navigator.max_retry_count = node["max_retry_count"].as<int>();
    return navigator;
}

GameConfig::CombatConfig parseCombatConfig(const YAML::Node& node) {
    GameConfig::CombatConfig combat;
    combat.ekf_default_dt = node["ekf_default_dt"].as<double>();
    combat.ekf_max_dt = node["ekf_max_dt"].as<double>();
    combat.debug_log_interval = node["debug_log_interval"].as<int>();
    return combat;
}

GameConfig::WaypointConfig parseWaypointConfig(const YAML::Node& node) {
    GameConfig::WaypointConfig waypoint;
    waypoint.health_ratio_escape = node["health_ratio_escape"].as<double>();
    waypoint.health_change_fallback = node["health_change_fallback"].as<int>();
    return waypoint;
}

}  // namespace
