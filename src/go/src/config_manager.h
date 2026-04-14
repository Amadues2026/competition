/**
 * @file config_manager.h
 * @brief 配置管理器头文件
 * @details 提供从YAML文件加载游戏配置的功能
 */

#ifndef CONFIG_MANAGER_H
#define CONFIG_MANAGER_H

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include "waypoint.h"  // 包含Waypoint结构体定义

/**
 * @brief 游戏配置结构体
 * @details 存储完整的游戏配置，包括路径点、阈值等
 */
struct GameConfig {
    int player_id; ///< 玩家ID（1或2）
    std::vector<Waypoint> waypoints; ///< 路径点数组
    std::unordered_map<std::string, int> health_change_thresholds; ///< 血量变化阈值映射
    
    /**
     * @brief 速度配置结构体
     */
    struct SpeedConfig {
        double max_far;       ///< 远距离最大速度（地图单位/s）
        double max_near;      ///< 近距离最大速度（地图单位/s）
        double max_precise;   ///< 精确模式最大速度（地图单位/s）
        double switch_distance;         ///< 速度切换距离（地图单位）
        double switch_distance_fast;   ///< 快速模式速度切换距离
        double precise_mode_distance;
        double arrival_threshold_fast;
        double arrival_threshold_normal;
        double precise_arrival_threshold;
        double precise_adjust_speed;
    } speed;
    
    // 相机参数
    struct CameraConfig {
        double fx;
        double fy;
        double cx;
        double cy;
    } camera;
    
    // 装甲板参数
    struct ArmorConfig {
        double width;
        double height;
    } armor;
    
    // 弹道参数
    struct BallisticsConfig {
        double bullet_speed;
        double gravity;
    } ballistics;

    // 位置调整参数
    struct PositionAdjustConfig {
        double hold_threshold;
        double adjust_interval;
    } position_adjust;

    // 摆动参数
    struct SwingConfig {
        double range;
        double speed;
    } swing;

    // 检测失败参数
    struct DetectionFailureConfig {
        int max_fail_count;
    } detection_failure;

    // 超时参数
    struct TimeoutConfig {
        double move;
        double health_history;
        double other_data;
    } timeout;

    // 射击阈值
    struct ThresholdConfig {
        int consecutive_miss_disable;
        int consecutive_miss_swing;
        int hit;
    } threshold;

    // 滤波器参数
    struct FilterConfig {
        int distance_window;
        double yaw_alpha;
        double pitch_alpha;
    } filter;

    // 单位换算
    struct UnitConfig {
        double map_unit_to_meter;
        double meter_to_map_unit;
    } unit;
};

/**
 * @brief 配置管理器类
 * @details 负责从YAML文件加载和管理游戏配置
 */
class ConfigManager {
private:
    GameConfig config_; ///< 游戏配置
    bool loaded_;      ///< 配置是否已加载标志

public:
    /**
     * @brief 构造函数
     */
    ConfigManager() : loaded_(false) {}
    
    /**
     * @brief 从YAML文件加载配置
     * @param config_path YAML配置文件路径
     * @return 如果加载成功返回true，否则返回false
     * @note 加载失败时会输出错误日志
     */
    bool loadFromFile(const std::string& config_path);
    
    /**
     * @brief 获取游戏配置
     * @return 游戏配置的常量引用
     * @note 如果配置未加载，行为未定义
     */
    const GameConfig& getConfig() const { return config_; }
    
    /**
     * @brief 检查配置是否已加载
     * @return 如果已加载返回true，否则返回false
     */
    bool isLoaded() const { return loaded_; }
    
private:
    // 解析路径点
    std::vector<Waypoint> parseWaypoints(const YAML::Node& node);
    
    // 解析血量变化阈值
    std::unordered_map<std::string, int> parseHealthThresholds(const YAML::Node& node);
    
    // 解析速度配置
    GameConfig::SpeedConfig parseSpeedConfig(const YAML::Node& node);
    
    // 解析相机配置
    GameConfig::CameraConfig parseCameraConfig(const YAML::Node& node);
    
    // 解析装甲板配置
    GameConfig::ArmorConfig parseArmorConfig(const YAML::Node& node);
    
    // 解析弹道配置
    GameConfig::BallisticsConfig parseBallisticsConfig(const YAML::Node& node);

    // 解析位置调整配置
    GameConfig::PositionAdjustConfig parsePositionAdjustConfig(const YAML::Node& node);

    // 解析摆动配置
    GameConfig::SwingConfig parseSwingConfig(const YAML::Node& node);

    // 解析检测失败配置
    GameConfig::DetectionFailureConfig parseDetectionFailureConfig(const YAML::Node& node);

    // 解析超时配置
    GameConfig::TimeoutConfig parseTimeoutConfig(const YAML::Node& node);

    // 解析射击阈值配置
    GameConfig::ThresholdConfig parseThresholdConfig(const YAML::Node& node);

    // 解析滤波器配置
    GameConfig::FilterConfig parseFilterConfig(const YAML::Node& node);

    // 解析单位换算配置
    GameConfig::UnitConfig parseUnitConfig(const YAML::Node& node);

    // 获取配置（带默认值）
    template<typename T>
    T getOrDefault(const YAML::Node& node, const std::string& key, const T& default_val) const;
};

#endif // CONFIG_MANAGER_H