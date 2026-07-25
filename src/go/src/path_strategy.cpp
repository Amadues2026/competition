/**
 * @file path_strategy.cpp
 * @brief 路径策略定义 —— 根据 YAML 路径点生成固定访问顺序
 *
 * 路径点坐标、云台指向和射击任务由 ConfigManager 从 game_config.yaml 解析。
 * 本文件只负责路径访问顺序，不感知阵营，也不保存重复的路径配置。
 *
 * 当前 FixedPathStrategy 按 YAML 顺序访问全部路径点：
 *   [0, 1, 2, ..., N-1]
 *
 * 路径循环、基地环路等索引约定由 PathNavigator 和 waypoint_utils 管理。
 * 修改路径点数量时，相关循环索引约定仍需同步检查。
 */

#include "path_strategy.h"

#include <numeric>
#include <stdexcept>

/**
 * @brief FixedPathStrategy 构造 —— 仅保存 GoNode 中的路径点数量
 */
FixedPathStrategy::FixedPathStrategy(size_t waypoint_count) : waypoint_count_(waypoint_count) {}

/**
 * @brief 生成路径索引序列（顺序遍历所有路径点）
 * @return [0, 1, 2, ..., waypoint_count_-1]
 */
std::vector<size_t> FixedPathStrategy::generatePathIndices() const {
    std::vector<size_t> indices(waypoint_count_);
    std::iota(indices.begin(), indices.end(), size_t{0});
    return indices;
}

/**
 * @brief 根据路径点数量创建固定路径策略
 * @throws std::invalid_argument 路径点为空
 */
std::unique_ptr<IPathPlanningStrategy> PathStrategyFactory::createStrategy(size_t waypoint_count) {
    if (waypoint_count == 0) {
        throw std::invalid_argument("Waypoints cannot be empty");
    }
    return std::make_unique<FixedPathStrategy>(waypoint_count);
}
