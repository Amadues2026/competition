#ifndef COMPETITION_DETECTION_TYPES_H
#define COMPETITION_DETECTION_TYPES_H

#include <opencv2/core/types.hpp>

/** @brief 装甲板检测结果，由检测、决策和可视化模块共享。 */
struct DetectedArmor {
    cv::Point2f center;     ///< 检测框中心点（像素坐标）
    cv::Rect armorBbox;     ///< 边界框（像素坐标）
    float score;            ///< 置信度
    int class_id;           ///< 类别ID
    float distance = 0.0f;  ///< PnP 解算出的距离（地图坐标单位）
};

#endif
