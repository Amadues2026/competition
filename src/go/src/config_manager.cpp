#include "config_manager.h"
#include <iostream>
#include <stdexcept>

bool ConfigManager::loadFromFile(const std::string& config_path) {
    try {
        YAML::Node config = YAML::LoadFile(config_path);
        
        // 加载玩家ID
        config_.player_id = config["player"]["id"].as<int>();
        
        // 根据玩家ID加载对应的路径点
        std::string waypoint_key = (config_.player_id == 1) ? "player1_waypoints" : "player2_waypoints";
        config_.waypoints = parseWaypoints(config[waypoint_key]);
        
        // 加载血量变化阈值
        config_.health_change_thresholds = parseHealthThresholds(config["health_change_thresholds"]);
        
        // 加载速度配置
        config_.speed = parseSpeedConfig(config["speed"]);
        
        // 加载相机配置
        config_.camera = parseCameraConfig(config["camera"]);
        
        // 加载装甲板配置
                config_.armor = parseArmorConfig(config["armor"]);
        
                // 加载弹道配置
                config_.ballistics = parseBallisticsConfig(config["ballistics"]);
        
                // 加载位置调整配置
                config_.position_adjust = parsePositionAdjustConfig(config["position_adjust"]);
        
                // 加载摆动配置
                config_.swing = parseSwingConfig(config["swing"]);
        
                // 加载检测失败配置
                config_.detection_failure = parseDetectionFailureConfig(config["detection_failure"]);
        
                // 加载超时配置
                config_.timeout = parseTimeoutConfig(config["timeout"]);
        
                // 加载射击阈值配置
                config_.threshold = parseThresholdConfig(config["threshold"]);
        
                // 加载滤波器配置
                config_.filter = parseFilterConfig(config["filter"]);
        
                // 加载单位换算配置
                config_.unit = parseUnitConfig(config["unit"]);        
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

std::vector<Waypoint> ConfigManager::parseWaypoints(const YAML::Node& node) {
    std::vector<Waypoint> waypoints;
    
    if (!node.IsSequence()) {
        throw std::runtime_error("路径点配置必须是序列");
    }
    
    for (const auto& wp_node : node) {
        Waypoint wp;
        wp.x = wp_node["x"].as<double>();
        wp.y = wp_node["y"].as<double>();
        wp.yaw = wp_node["yaw"].as<double>();
        wp.pitch = wp_node["pitch"].as<double>();
        wp.has_shoot_task = wp_node["has_shoot_task"].as<bool>();
        waypoints.push_back(wp);
    }
    
    return waypoints;
}

std::unordered_map<std::string, int> ConfigManager::parseHealthThresholds(const YAML::Node& node) {
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

GameConfig::SpeedConfig ConfigManager::parseSpeedConfig(const YAML::Node& node) {
    GameConfig::SpeedConfig speed;
    speed.max_far = node["max_far"].as<double>();
    speed.max_near = node["max_near"].as<double>();
    speed.max_precise = node["max_precise"].as<double>();
    speed.switch_distance = node["switch_distance"].as<double>();
    speed.switch_distance_fast = node["switch_distance_fast"].as<double>();
    speed.precise_mode_distance = node["precise_mode_distance"].as<double>();
    speed.arrival_threshold_fast = node["arrival_threshold_fast"].as<double>();
    speed.arrival_threshold_normal = node["arrival_threshold_normal"].as<double>();
    speed.precise_arrival_threshold = node["precise_arrival_threshold"].as<double>();
    speed.precise_adjust_speed = node["precise_adjust_speed"].as<double>();
    return speed;
}

GameConfig::CameraConfig ConfigManager::parseCameraConfig(const YAML::Node& node) {
    GameConfig::CameraConfig camera;
    camera.fx = node["fx"].as<double>();
    camera.fy = node["fy"].as<double>();
    camera.cx = node["cx"].as<double>();
    camera.cy = node["cy"].as<double>();
    return camera;
}

GameConfig::ArmorConfig ConfigManager::parseArmorConfig(const YAML::Node& node) {
    GameConfig::ArmorConfig armor;
    armor.width = node["width"].as<double>();
    armor.height = node["height"].as<double>();
    return armor;
}

GameConfig::BallisticsConfig ConfigManager::parseBallisticsConfig(const YAML::Node& node) {
    GameConfig::BallisticsConfig ballistics;
    ballistics.bullet_speed = node["bullet_speed"].as<double>();
    ballistics.gravity = node["gravity"].as<double>();
    return ballistics;
}

GameConfig::PositionAdjustConfig ConfigManager::parsePositionAdjustConfig(const YAML::Node& node) {
    GameConfig::PositionAdjustConfig position_adjust;
    position_adjust.hold_threshold = node["hold_threshold"].as<double>();
    position_adjust.adjust_interval = node["adjust_interval"].as<double>();
    return position_adjust;
}

GameConfig::SwingConfig ConfigManager::parseSwingConfig(const YAML::Node& node) {
    GameConfig::SwingConfig swing;
    swing.range = node["range"].as<double>();
    swing.speed = node["speed"].as<double>();
    return swing;
}

GameConfig::DetectionFailureConfig ConfigManager::parseDetectionFailureConfig(const YAML::Node& node) {
    GameConfig::DetectionFailureConfig detection_failure;
    detection_failure.max_fail_count = node["max_fail_count"].as<int>();
    return detection_failure;
}

GameConfig::TimeoutConfig ConfigManager::parseTimeoutConfig(const YAML::Node& node) {
    GameConfig::TimeoutConfig timeout;
    timeout.move = node["move"].as<double>();
    timeout.health_history = node["health_history"].as<double>();
    timeout.other_data = node["other_data"].as<double>();
    return timeout;
}

GameConfig::ThresholdConfig ConfigManager::parseThresholdConfig(const YAML::Node& node) {
    GameConfig::ThresholdConfig threshold;
    threshold.consecutive_miss_disable = node["consecutive_miss_disable"].as<int>();
    threshold.consecutive_miss_swing = node["consecutive_miss_swing"].as<int>();
    threshold.hit = node["hit"].as<int>();
    return threshold;
}

GameConfig::FilterConfig ConfigManager::parseFilterConfig(const YAML::Node& node) {
    GameConfig::FilterConfig filter;
    filter.distance_window = node["distance_window"].as<int>();
    filter.yaw_alpha = node["yaw_alpha"].as<double>();
    filter.pitch_alpha = node["pitch_alpha"].as<double>();
    return filter;
}

GameConfig::UnitConfig ConfigManager::parseUnitConfig(const YAML::Node& node) {
    GameConfig::UnitConfig unit;
    unit.map_unit_to_meter = node["map_unit_to_meter"].as<double>();
    unit.meter_to_map_unit = node["meter_to_map_unit"].as<double>();
    return unit;
}