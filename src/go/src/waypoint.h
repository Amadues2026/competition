/**
 * @file waypoint.h
 * @brief 路径点结构体定义
 * @details Waypoint结构体在多个模块中使用，统一在此定义
 */

#ifndef WAYPOINT_H
#define WAYPOINT_H

/**
 * @brief 路径点结构体
 * @details 存储单个路径点的位置、角度和任务信息
 */
struct Waypoint {
    double x;           ///< X坐标（米）
    double y;           ///< Y坐标（米）
    double yaw;         ///< 云台偏航角（度）
    double pitch;       ///< 云台俯仰角（度）
    bool has_shoot_task; ///< 是否有射击任务
};

#endif  // WAYPOINT_H