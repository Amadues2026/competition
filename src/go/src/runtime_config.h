/**
 * @file runtime_config.h
 * @brief 运行时配置结构体 —— 存储从 YAML 配置文件加载的所有运行参数
 *
 * RuntimeConfig 是 GameConfig（YAML 原始配置）经过单位换算后的运行时版本。
 * 所有距离参数已从"米"转换为"地图坐标单位"（乘以 meter_to_map_unit）。
 *
 * 与 GameConfig 的区别：
 *   - GameConfig 存储 YAML 原始值（人类可读，单位为米）
 *   - RuntimeConfig 存储运行时值（代码直接使用，单位为地图坐标）
 *   - GoNode::loadConfigParameters() 负责转换和赋值
 *
 * 参数分组：
 *   速度控制：max_speed_far/near, speed_switch_distance, precise_mode_distance
 *   到达判定：arrival_threshold_fast/normal/precise_arrival_threshold
 *   位置保持：position_hold_threshold, position_adjust_interval
 *   数据保留：health_history_duration, other_data_duration
 *   检测容错：detection_fail_max
 *
 * 默认值设计原则：
 *   - 偏保守（宁可慢一点也不要撞墙/错过目标）
 *   - 速度默认值允许在 YAML 加载失败时仍能正常运行
 */

#ifndef COMPETITION_RUNTIME_CONFIG_H
#define COMPETITION_RUNTIME_CONFIG_H

struct RuntimeConfig {
    // ── 速度控制参数 ──
    double max_speed_far = 50.0;               ///< 远距离最大速度（地图单位/秒）
    double max_speed_near = 15.0;              ///< 近距离最大速度（地图单位/秒）
    double speed_switch_distance = 5.0;        ///< 远→近速度切换距离阈值
    double speed_switch_distance_fast = 10.0;  ///< 快速模式下的切换阈值（更保守）
    double precise_mode_distance = 1.5;        ///< 精调模式距离阈值（进入后大幅降速）

    // ── 到达判定参数（三档递减） ──
    double arrival_threshold_fast = 2.0;     ///< 快速模式到达阈值（粗略到达即可）
    double arrival_threshold_normal = 0.8;   ///< 正常模式到达阈值
    double precise_arrival_threshold = 0.2;  ///< 精调模式到达阈值（高精度停靠）
    double precise_adjust_speed = 1.5;       ///< 精调模式下的移动速度

    // ── 位置保持参数 ──
    double position_hold_threshold = 0.05;  ///< 射击位置保持的位移容差
    double position_adjust_interval = 1.0;  ///< 位置保持调用间隔（秒）

    // ── 数据保留时长 ──
    double health_history_duration = 5.0;  ///< 血量历史保留时长（秒）
    double other_data_duration = 10.0;     ///< 位置/角度历史保留时长（秒）

    // ── 检测容错 ──
    int detection_fail_max = 10;       ///< 连续检测失败次数上限
    bool enable_visualization = true;  ///< 是否启用 GUI 显示窗口（无头服务器建议关闭）

    // ── 距离门控 ──
    double default_detection_distance = 3.5;  ///< 默认检测距离（无检测时回退）
    double distance_gate_max = 30.0;          ///< 距离门控上限（超过此值丢弃）
    double distance_gate_acceptance = 10.0;   ///< 帧间距离跳变容忍度

    // ── EKF 参数 ──
    double ekf_q_yaw = 0.01;                 ///< 偏航角过程噪声
    double ekf_q_pitch = 0.01;               ///< 俯仰角过程噪声
    double ekf_q_distance = 0.5;             ///< 距离过程噪声
    double ekf_r_yaw = 4e-3;                 ///< 偏航角测量噪声
    double ekf_r_pitch = 4e-3;               ///< 俯仰角测量噪声
    double ekf_r_distance_denom = 200.0;     ///< 距离自适应噪声分母
    double ekf_r_distance_base = 0.09;       ///< 距离自适应噪声基值
    double ekf_p0_yaw = 1e-2;                ///< 初始协方差 - 偏航
    double ekf_p0_pitch = 1e-2;              ///< 初始协方差 - 俯仰
    double ekf_p0_distance = 1e-1;           ///< 初始协方差 - 距离
    double ekf_p_reset = 1e3;                ///< 重置时协方差倍率
    double ekf_q_adaptive_alpha = 0.0;       ///< 自适应Q系数（0=禁用）
    double ekf_q_adaptive_max_scale = 10.0;  ///< 自适应Q最大倍率

    // ── NIS 参数 ──
    double nis_chi2_upper = 7.815;  ///< NIS 卡方检验上界（自由度3，95%分位）
    int nis_window = 100;           ///< NIS 失败率统计窗口

    // ── 跟踪状态机参数 ──
    int tracking_min_detect = 3;            ///< 确认检测所需最少帧数
    int tracking_max_temp_lost = 15;        ///< 临时丢失最大容忍帧数
    double tracking_nis_failure_max = 0.3;  ///< NIS 失败率阈值

    // ── 检测参数 ──
    double detection_conf_threshold = 0.1;  ///< YOLO 置信度阈值
    double detection_nms_iou = 0.45;        ///< NMS IoU 阈值

    // ── 定时器间隔（毫秒） ──
    int timer_control = 5;          ///< 运动控制间隔
    int timer_shoot = 30;           ///< 自瞄检测间隔
    int timer_health_check = 1000;  ///< 血量监控间隔
    int timer_swing = 20;           ///< 摆动+射击间隔
    int timer_visualization = 67;   ///< GUI 刷新间隔（独立线程）

    // ── 漂移检测 ──
    double drift_threshold_multiplier = 2.0;  ///< 漂移判定乘数
    double drift_timeout = 5.0;               ///< 漂移超时（秒）

    // ── 目标锁定加速倍率 ──
    double lock_speed_multiplier = 2.0;  ///< 锁定时变速区间扩大倍率

    // ── 数据管理器 ──
    int data_max_health_change = 500;    ///< 单次血量变化上限
    double data_health_tolerance = 0.1;  ///< 血量历史窗口容差（秒）
    int data_max_health_changes = 10;    ///< 血量变化记录最大条数
    double data_yaw_lookback = 1.0;      ///< 云台角度回溯时间（秒）

    // ── 导航器 ──
    int navigator_max_retry = 3;  ///< 卡住重试次数上限

    // ── 战斗管理器 ──
    double combat_ekf_default_dt = 0.03;  ///< EKF 默认时间间隔（秒）
    double combat_ekf_max_dt = 0.5;       ///< EKF 最大时间间隔（秒）
    int combat_debug_log_interval = 30;   ///< 调试日志间隔（帧）

    // ── 路点工具 ──
    double waypoint_health_ratio_escape = 0.2;  ///< 血量比例逃逸阈值
    int waypoint_health_change_fallback = 50;   ///< 血量变化阈值回退值
};

#endif
