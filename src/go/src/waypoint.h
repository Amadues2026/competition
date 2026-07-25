/**
 * @file waypoint.h
 * @brief 路径点数据结构定义 —— 存储单个路径点的位置、云台指向、射击任务标记
 *
 * Waypoint 是整个导航和射击系统的基础数据单元。
 * 路径点序列由 PathStrategy 生成，存储在 GoNode 中，被 MovementController/CombatManager 等读取。
 *
 * 字段说明：
 *   x, y           - 世界坐标系位置（地图坐标，范围 [-50, 50]，原点在地图中心）
 *   yaw            - 到达后云台水平指向角（度，0=正前方，正值=逆时针）
 *   pitch          - 到达后云台俯仰角（度，正值=上仰，负值=下压）
 *   has_shoot_task - 是否执行射击任务
 *                    true  = 到达后进入自瞄检测、弹道补偿、开火流程
 *                    false = 纯移动点（如出发段 P1-P3），到达后直接切换下一个点
 *
 * 坐标系约定：
 *   - 世界坐标系固定，与 player_id 无关
 *   - 蓝方出生在 (-x, -y) 区域，红方出生在 (+x, +y) 区域
 *   - 路径点坐标经过实机调参验证，修改后需要重新测试
 */

#ifndef WAYPOINT_H
#define WAYPOINT_H

struct Waypoint {
    double x;              ///< 世界坐标 X
    double y;              ///< 世界坐标 Y
    double yaw;            ///< 到达后云台水平指向角（度）
    double pitch;          ///< 到达后云台俯仰角（度）
    bool has_shoot_task;   ///< 是否执行射击任务（true=进入自瞄流程）
};

#endif
