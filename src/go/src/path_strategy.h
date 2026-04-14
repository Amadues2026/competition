#ifndef PATH_STRATEGY_H
#define PATH_STRATEGY_H

#include <vector>
#include <memory>
#include "waypoint.h"  // 包含Waypoint结构体定义（统一在waypoint.h中定义，避免重复定义）

// 路径规划策略接口（策略模式）
class IPathPlanningStrategy {
public:
    virtual ~IPathPlanningStrategy() = default;
    
    // 生成路径点序列
    virtual std::vector<size_t> generatePathIndices() = 0;
    
    // 获取路径点数组
    virtual const std::vector<Waypoint>& getWaypoints() = 0;
    
    // 获取策略名称
    virtual std::string getStrategyName() const = 0;
};

// Player1路径策略（蓝方）
class Player1PathStrategy : public IPathPlanningStrategy {
private:
    std::vector<Waypoint> waypoints_;
    
public:
    Player1PathStrategy();
    std::vector<size_t> generatePathIndices() override;
    const std::vector<Waypoint>& getWaypoints() override;
    std::string getStrategyName() const override;
};

// Player2路径策略（红方）
class Player2PathStrategy : public IPathPlanningStrategy {
private:
    std::vector<Waypoint> waypoints_;
    
public:
    Player2PathStrategy();
    std::vector<size_t> generatePathIndices() override;
    const std::vector<Waypoint>& getWaypoints() override;
    std::string getStrategyName() const override;
};

// 路径规划策略工厂
class PathStrategyFactory {
public:
    static std::unique_ptr<IPathPlanningStrategy> createStrategy(int player_id);
};

#endif // PATH_STRATEGY_H