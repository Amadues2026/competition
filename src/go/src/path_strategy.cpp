#include "path_strategy.h"
#include <stdexcept>

// Player1路径点（蓝方）
Player1PathStrategy::Player1PathStrategy() {
    waypoints_ = {
        {15.9, -44.9, 0.0, 0.0, false},  // P4 - 敌方前哨站
        {5.8, -44.9, 0.0, 0.0, false},   // P5 - 补给站
        {5.8, -37.8, 0.0, 0.0, false},   // P6 - 进攻点
        {-27.7, -38.3, 243.0, -1.5, true},   // 攻击5号敌人
        {-27.2, -2.4, 89.5, 2.5, true},      // 攻击4号敌人
        {-2.5, -2.8, 58.7, 19.5, true},      // 攻击前哨站
        {-3.0, 32.0, 112.8, 8.0, true},      // 敌方基地
        {6.8, 32.0, 48.5, 8.0, true},        // 敌方基地
        {6.8, 47.5, 314.0, 8.0, true},       // 敌方基地
        {-6.0, 47.5, 221.0, 8.0, true},      // P10 - 攻击基地
        {-6.0, 32.0, 123.0, 8.0, true}       // P11 - 攻击基地
    };
}

std::vector<size_t> Player1PathStrategy::generatePathIndices() {
    // Player1路径：P1-P11顺序
    std::vector<size_t> indices;
    for (size_t i = 0; i < waypoints_.size(); i++) {
        indices.push_back(i);
    }
    return indices;
}

const std::vector<Waypoint>& Player1PathStrategy::getWaypoints() {
    return waypoints_;
}

std::string Player1PathStrategy::getStrategyName() const {
    return "Player1PathStrategy";
}

// Player2路径点（红方，坐标中心对称）
Player2PathStrategy::Player2PathStrategy() {
    waypoints_ = {
        {-15.9, 44.9, 0.0, 0.0, false},   // P4 - 敌方前哨站
        {-5.8, 44.9, 0.0, 0.0, false},    // P5 - 补给站
        {-5.8, 37.8, 0.0, 0.0, false},    // P6 - 进攻点
        {27.7, 38.3, 243.0, -1.5, true},  // 攻击5号敌人
        {27.2, 2.4, 89.5, 2.5, true},     // 攻击4号敌人
        {2.5, 2.8, 58.7, 19.5, true},     // 攻击前哨站
        {3.0, -32.0, 112.8, 8.0, true},   // 敌方基地
        {-6.8, -32.0, 48.5, 8.0, true},   // 敌方基地
        {-6.8, -47.5, 314.0, 8.0, true},  // 敌方基地
        {6.0, -47.5, 221.0, 8.0, true},   // P10 - 攻击基地
        {6.0, -32.0, 123.0, 8.0, true}    // P11 - 攻击基地
    };
}

std::vector<size_t> Player2PathStrategy::generatePathIndices() {
    // Player2路径：P1-P11顺序
    std::vector<size_t> indices;
    for (size_t i = 0; i < waypoints_.size(); i++) {
        indices.push_back(i);
    }
    return indices;
}

const std::vector<Waypoint>& Player2PathStrategy::getWaypoints() {
    return waypoints_;
}

std::string Player2PathStrategy::getStrategyName() const {
    return "Player2PathStrategy";
}

// 工厂方法
std::unique_ptr<IPathPlanningStrategy> PathStrategyFactory::createStrategy(int player_id) {
    if (player_id == 1) {
        return std::make_unique<Player1PathStrategy>();
    } else if (player_id == 2) {
        return std::make_unique<Player2PathStrategy>();
    } else {
        throw std::invalid_argument("Invalid player_id: " + std::to_string(player_id));
    }
}