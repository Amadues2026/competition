/**
 * @file path_strategy.h
 * @brief 路径策略接口和工厂声明 —— 定义路径生成的抽象接口和固定路径实现
 *
 * 采用策略模式 + 工厂模式：
 *   - IPathPlanningStrategy: 抽象接口，定义 generatePathIndices()
 *   - FixedPathStrategy: 根据航点数量生成固定访问顺序
 *   - PathStrategyFactory: 工厂类，根据配置路径点创建策略实例
 *
 * 扩展性：
 *   如果未来需要动态路径规划（如 A* 寻路），只需实现新的 IPathPlanningStrategy 子类，
 *   并在 PathStrategyFactory 中注册即可，不影响其他模块。
 *
 * 当前实现：
 *   只有 FixedPathStrategy（固定顺序），路径点坐标以 game_config.yaml 为唯一数据源。
 */

#ifndef PATH_STRATEGY_H
#define PATH_STRATEGY_H

#include <cstddef>
#include <memory>
#include <vector>

/**
 * @brief 路径规划策略抽象接口
 *
 * 路径策略只生成访问索引，不持有路径点数据。
 */
class IPathPlanningStrategy {
public:
    virtual ~IPathPlanningStrategy() = default;

    /**
     * @brief 生成路径点访问索引序列
     * @return 索引数组，映射到 GoNode 持有的路径点
     *
     * 例如：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 表示按顺序访问所有路径点
     */
    virtual std::vector<size_t> generatePathIndices() const = 0;
};

/**
 * @brief 固定路径策略 —— 按 YAML 给定顺序访问路径点
 *
 * 航点数量由 GoNode 注入，策略不持有坐标，也不感知阵营。
 * generatePathIndices() 返回简单的线性序列 [0, 1, 2, ...]。
 */
class FixedPathStrategy : public IPathPlanningStrategy {
private:
    size_t waypoint_count_;  ///< GoNode 持有的路径点数量

public:
    /** @brief 构造函数（仅保存路径点数量） */
    explicit FixedPathStrategy(size_t waypoint_count);

    /** @brief 生成线性索引序列 [0, 1, ..., N-1] */
    std::vector<size_t> generatePathIndices() const override;
};

/**
 * @brief 路径策略工厂 —— 根据配置路径点创建策略实例
 *
 * 使用方式：
 *   auto strategy = PathStrategyFactory::createStrategy(waypoints.size());
 */
class PathStrategyFactory {
public:
    /**
     * @brief 创建路径策略
     * @param waypoint_count GoNode 持有的路径点数量
     * @return 策略实例（unique_ptr，调用方接管所有权）
     * @throws std::invalid_argument 路径点为空
     */
    static std::unique_ptr<IPathPlanningStrategy> createStrategy(size_t waypoint_count);
};

#endif
