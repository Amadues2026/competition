/**
 * @file config_manager.h
 * @brief YAML 配置文件解析器声明 —— GameConfig 结构体 + ConfigManager 解析类
 *
 * GameConfig 是所有运行参数的强类型容器，嵌套结构体对应 YAML 中的各配置分类。
 * ConfigManager 负责从 YAML 文件加载并填充 GameConfig。
 *
 * 设计决策：
 *   - 使用嵌套结构体而非扁平 map，编译期类型安全
 *   - 普通运行参数在 RuntimeConfig 中有默认值；路径点无代码副本，YAML 失败时拒绝启动
 *   - health_change_thresholds 使用 unordered_map（键不固定，按路径点名称索引）
 */

#ifndef CONFIG_MANAGER_H
#define CONFIG_MANAGER_H

#include <string>
#include <unordered_map>
#include <vector>

#include "waypoint.h"

/**
 * @brief 游戏配置结构体 —— 存储所有从 YAML 加载的运行参数
 *
 * 子结构体对应 game_config.yaml 中的各顶层 key：
 *   speed, camera, armor, ballistics, position_adjust,
 *   swing, detection_failure, timeout, threshold, filter, unit,
 *   ekf, nis, tracking, detection, timer, drift, lock, data,
 *   player1_waypoints, player2_waypoints, navigator, combat, waypoint
 */
struct GameConfig {
    int player_id;                                                  ///< 玩家ID（1/2）
    std::vector<Waypoint> player1_waypoints;                        ///< 蓝方路径点（YAML 唯一数据源）
    std::vector<Waypoint> player2_waypoints;                        ///< 红方路径点（YAML 唯一数据源）
    std::unordered_map<std::string, int> health_change_thresholds;  ///< 血量变化阈值（按路径点名称索引）

    /** @brief 运动控制速度参数（单位：米，运行时需转换为地图坐标） */
    struct SpeedConfig {
        double max_far;                    ///< 远距离最大速度
        double max_near;                   ///< 近距离最大速度
        double switch_distance;            ///< 远→近切换距离
        double switch_distance_fast;       ///< 快速模式切换距离
        double precise_mode_distance;      ///< 精调模式距离阈值
        double arrival_threshold_fast;     ///< 快速到达阈值
        double arrival_threshold_normal;   ///< 正常到达阈值
        double precise_arrival_threshold;  ///< 精调到达阈值
        double precise_adjust_speed;       ///< 精调移动速度
    } speed;

    struct CameraConfig {
        double fx, fy, cx, cy;
    } camera;  ///< 相机内参（PnP 用）
    struct ArmorConfig {
        double width, height;
    } armor;  ///< 装甲板尺寸（米）
    struct BallisticsConfig {
        double bullet_speed, gravity;
    } ballistics;  ///< 弹道参数
    struct PositionAdjustConfig {
        double hold_threshold, adjust_interval;
    } position_adjust;  ///< 位置保持
    struct SwingConfig {
        double range, speed;
    } swing;  ///< 摆动扫描参数
    struct DetectionFailureConfig {
        int max_fail_count;
    } detection_failure;  ///< 检测失败容忍
    struct TimeoutConfig {
        double move, health_history, other_data;
    } timeout;  ///< 超时配置
    struct ThresholdConfig {
        int consecutive_miss_disable, consecutive_miss_swing;
    } threshold;  ///< 未命中阈值
    struct DistanceGateConfig {
        double default_detection_distance, gate_max, gate_acceptance;
    } distance_gate;  ///< 距离门控
    struct FilterConfig {
        int distance_window;
    } filter;  ///< 滤波器窗口
    struct UnitConfig {
        double meter_to_map_unit;
    } unit;  ///< 单位换算系数

    // ── 新增配置分类（v8.7 将全部硬编码常量移入 YAML） ──
    struct EkfConfig {
        double q_yaw, q_pitch, q_distance, r_yaw, r_pitch, r_distance_denom, r_distance_base, p0_yaw, p0_pitch,
            p0_distance, p_reset, q_adaptive_alpha, q_adaptive_max_scale;
    } ekf;
    struct NisConfig {
        double chi2_upper;
        int window;
    } nis;
    struct TrackingConfig {
        int min_detect_count, max_temp_lost_count;
        double nis_failure_max;
    } tracking;
    struct DetectionConfig {
        double conf_threshold, nms_iou_threshold;
        bool enable_visualization;
    } detection;
    struct TimerConfig {
        int control, shoot, health_check, swing, visualization;
    } timer;
    struct DriftConfig {
        double threshold_multiplier, timeout;
    } drift;
    struct LockConfig {
        double speed_multiplier;
    } lock;
    struct DataConfig {
        int max_reasonable_health_change, max_health_changes;
        double health_history_tolerance, yaw_pitch_lookback;
    } data;
    struct NavigatorConfig {
        int max_retry_count;
    } navigator;
    struct CombatConfig {
        double ekf_default_dt, ekf_max_dt;
        int debug_log_interval;
    } combat;
    struct WaypointConfig {
        double health_ratio_escape;
        int health_change_fallback;
    } waypoint;
};

/**
 * @brief 配置管理器 —— 从 YAML 文件加载并解析为 GameConfig
 *
 * 使用方式：
 *   ConfigManager cm;
 *   if (cm.loadFromFile("game_config.yaml")) {
 *       const auto& cfg = cm.getConfig();
 *       // 使用 cfg.speed.max_far 等
 *   }
 */
class ConfigManager {
private:
    GameConfig config_;  ///< 解析后的配置数据
    bool loaded_;        ///< 是否已成功加载

public:
    ConfigManager() : loaded_(false) {}

    /** @brief 从 YAML 文件加载配置，返回是否成功 */
    bool loadFromFile(const std::string& config_path);

    /** @brief 获取配置数据的只读引用 */
    const GameConfig& getConfig() const { return config_; }

    /** @brief 检查配置是否已加载 */
    bool isLoaded() const { return loaded_; }
};

#endif
