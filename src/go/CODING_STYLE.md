# 代码风格规范

本文档描述项目中实际使用的代码风格，基于现有代码总结。

## 基本原则

- 使用4空格缩进，不使用Tab
- 每行尽量控制在120字符以内
- 使用Google风格的大括号（左大括号不换行）
- 指针和引用的`*`和`&`紧靠类型（左侧对齐）
- 注释使用中文，技术术语可保留英文

## 命名规范

### 类名
- 使用PascalCase（大驼峰）
- 示例：`TimeManager`, `PathNavigator`, `ShootController`, `DataManager`

### 函数名
- 使用camelCase（小驼峰）
- 示例：`initialize`, `updateGameTime`, `getCurrentTime`, `updateSelfHealth`

### 变量名
- 使用snake_case（下划线分隔）
- 示例：`game_start_time_`, `current_game_time_`, `player_id_`
- 成员变量以下划线结尾

### 常量名
- 使用UPPER_CASE（全大写+下划线）
- 示例：`MAX_SPEED_FAR`, `CAMERA_FX`, `MAX_RETRY_COUNT`, `MAP_UNIT_TO_METER`

### 文件名
- 头文件：snake_case.h
- 源文件：snake_case.cpp
- 示例：`config_manager.h`, `path_strategy.cpp`

## 注释风格

### 文件级注释
使用Doxygen `/** */` 风格，在文件最开头：
```cpp
/**
 * @file config_manager.cpp
 * @brief YAML 配置文件解析器 —— 将 game_config.yaml 加载为强类型 GameConfig 结构体
 *
 * ConfigManager 负责把用户可编辑的 YAML 配置转化为代码可直接使用的结构体。
 * 所有模块通过 config_manager.getConfig 获取配置值，避免硬编码常量分散各处。
 *
 * 支持的配置分类（对应 game_config.yaml 中的各顶层 key）：
 *   - speed:              运动控制参数（最大速度、距离阈值等，单位为米）
 *   - camera:             相机内参（fx, fy, cx, cy，用于 PnP 求解）
 *   - armor:              装甲板尺寸（宽/高，米，用于 PnP 求解）
 *
 * 错误处理策略：
 *   - YAML 解析异常 → 打日志返回 false，由调用方回退到代码默认值
 *   - 单个字段缺失 → 抛 runtime_error，外层 catch 统一处理
 */

#include "config_manager.h"
```

### 类级注释
使用 `/** @brief */` 风格，放在类定义之前：
```cpp
/**
 * @brief 目标扩展卡尔曼滤波器 —— 跟踪目标的 yaw/pitch/distance 三状态
 *
 * TargetEKF 封装 GeneralKalmanFilter，专门用于跟踪单个目标的云台指向和距离。
 * 状态向量 x = [yaw, pitch, distance]^T（3维）
 *
 * 使用场景：
 *   CombatManager 中的 target_ekf_ 实例，每帧调用 update 融合测量。
 */
class TargetEKF {
```

### 成员变量注释
使用 `///< description` 行尾 Doxygen 注释：
```cpp
struct SharedContext {
    int player_id = 0;                  ///< 玩家ID（1=蓝方，2=红方）
    double target_yaw = 0;              ///< 当前目标云台 yaw（度）

    std::atomic<bool> is_precise_adjusting{false};  ///< 是否处于精调模式

    struct ChassisLocked {
        State state = MOVING;                          ///< 当前工作状态
        bool is_under_attack = false;                  ///< 是否正在被攻击
        std::stack<size_t> attack_waypoint_stack;      ///< 被攻击时的路径点回退栈
    } chassis;
};
```

### 函数级注释
使用 `/** @brief ... @param ... @return ... */` 风格：
```cpp
/**
 * @brief 更新自身血量并计算血量变化量
 * @param health 当前血量
 * @param timestamp 当前时间戳（来自 ROS2 消息头）
 *
 * 血量变化量计算流程：
 *   1. 将新血量追加到 history 队列
 *   2. trimHistory 裁剪过期数据
 *   3. 等待 history 跨越完整一个 health_history_duration 周期
 *   4. 取最早一条记录的血量作为 "N秒前的血量"
 *   5. change = 当前血量 - N秒前的血量
 */
void DataManager::updateSelfHealth(int health, rclcpp::Time timestamp) {
```

### 常量注释
在常量定义处使用行尾 `///<` 或块注释说明：
```cpp
namespace waypoint_utils {
    /** @brief 敌方单位数量（玩家+3号+4号+5号+前哨站+基地 = 6个） */
    inline constexpr int kNumEnemies = 6;

    /** @brief 第一个射击路径点索引（P5=索引4），之前的点为纯移动 */
    inline constexpr size_t kFirstShootWaypointIndex = 4;

    /** @brief P4 路径点血量变化阈值（-50，负数表示"敌方血量减少超过此值"） */
    inline constexpr int HEALTH_CHANGE_THRESHOLD_P4 = -50;
}
```

### 枚举注释
在枚举定义前说明用途，枚举值使用行尾注释：
```cpp
/**
 * @brief 跟踪状态枚举
 *   LOST      - 未跟踪任何目标
 *   DETECTING - 检测到目标，等待连续确认
 *   TRACKING  - 稳定跟踪中（可开火）
 *   TEMP_LOST - 目标短暂丢失（等待找回）
 */
enum class TrackingState { LOST, DETECTING, TRACKING, TEMP_LOST };
```

### 结构体字段注释
使用行尾 `///<` 注释：
```cpp
/**
 * @brief 性能指标快照 —— 存储单个 metric 的统计结果
 */
struct PerformanceMetrics {
    std::string name;       ///< 指标名称
    double avg_ms;          ///< 平均执行时间（毫秒）
    double min_ms;          ///< 最小执行时间
    double max_ms;          ///< 最大执行时间
    double std_dev_ms;      ///< 标准差（反映波动程度）
    size_t sample_count;    ///< 总采样次数（不受滑动窗口影响）
};
```

### 分段注释
使用 `═══` 分隔符分隔逻辑段：
```cpp
// ══════════════════════════════════════════════════════════════
//  ROS2 回调函数 —— 接收模拟器数据，更新内部状态
// ══════════════════════════════════════════════════════════════

/**
 * @brief 位置回调 —— 更新机器人在世界坐标系中的 (x, y)
 */
void GoNode::positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
```

### 行内注释
对于非显而易见的逻辑，使用行内 `//` 注释：
```cpp
// 优先从配置文件读取
if (config && config->isLoaded) {
    const auto& thresholds = config->getConfig.health_change_thresholds;
    // ...
}
// 回退到代码默认常量
switch (waypoint_idx) {
```

## 格式规范

### 大括号
使用Google风格（左大括号不换行）：
```cpp
if (condition) {
    // 代码
} else {
    // 代码
}

class MyClass {
public:
    // 代码
};
```

### 空格
- 运算符前后加空格：`a = b + c`
- 逗号后加空格：`func(a, b, c)`
- 逗号前不加空格
- 控制语句关键字后加空格：`if (condition)`, `for (int i = 0; i < n; i++)`

### 空行
- 函数定义之间空1行
- 类的成员变量之间空1行（或按逻辑分组空行）
- 逻辑块之间空1行（如果有助于可读性）

### 指针和引用
星号和`&`紧靠类型（左侧对齐）：
```cpp
int* ptr = nullptr;
const std::string& ref = str;
std::unique_ptr<CenterNetDetector> centernet_detector_;
```

## 头文件包含顺序

按照以下顺序组织：
1. 当前 `.cpp` 对应的同名头文件（仅源文件，例如 `foo.cpp` 先包含 `foo.h`）
2. 第三方库（TensorRT、CUDA、yaml-cpp、GTest 等）
3. C++标准库
4. ROS2头文件
5. OpenCV头文件
6. 其他本地头文件

`.clang-format` 会自动排序并用空行分组，不会自动生成分组标题；标题注释可选。示例：
```cpp
// 当前源文件对应头文件
#include "yolov5_detector.h"

// ==================== 第三方库 ====================
#include <NvInfer.h>
#include <cuda_runtime_api.h>

// ==================== C++标准库 ====================
#include <cmath>
#include <vector>
#include <memory>

// ==================== ROS2核心头文件 ====================
#include "rclcpp/rclcpp.hpp"

// ==================== ROS2消息类型头文件 ====================
#include "geometry_msgs/msg/twist_stamped.hpp"

// ==================== OpenCV头文件 ====================
#include <opencv2/opencv.hpp>

// ==================== 本地头文件 ====================
#include "config_manager.h"
#include "waypoint.h"
```

## 代码组织

### 类成员顺序
1. 常量定义（`static constexpr`）
2. 私有成员变量
3. 公共成员函数
4. 私有成员函数
5. 构造函数

### 函数组织
- 简短函数（<20）可以写在类定义中
- 复杂函数（>20行）应在类外实现，头文件中只留声明
- 相关函数放在一起

## 特殊规范

### 原子操作
对于多线程访问的简单类型，使用`std::atomic`：
```cpp
std::atomic<int> atomic_self_health_;

// 写操作（无锁，200Hz 安全）
atomic_self_health_.store(health, std::memory_order_relaxed);

// 读操作（无锁，200Hz 安全）
int health = atomic_self_health_.load(std::memory_order_relaxed);
```

### 线程安全
使用 `std::mutex` 保护共享数据：
```cpp
class DataManager {
    mutable std::mutex mutex_;  ///< 互斥锁

    void updateSelfHealth(int health, rclcpp::Time timestamp) {
        std::lock_guard<std::mutex> lock(mutex_);
        // ... 安全修改共享数据
    }
};
```

### 异常处理
使用`try-catch`捕获异常，使用`RCLCPP_ERROR`记录错误：
```cpp
try {
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data, size));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize CUDA engine");
    }
} catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger, "TensorRT初始化失败：%s", e.what);
    throw;  // 重新抛出
}
```

### CUDA错误处理
检查CUDA函数返回值：
```cpp
cudaError_t err = cudaMalloc(&gpu_buffers_[0], input_size_);
if (err != cudaSuccess) {
    RCLCPP_ERROR(this->get_logger, "CUDA内存分配失败：%s",
                 cudaGetErrorString(err));
    throw std::runtime_error("CUDA内存分配失败");
}
```

### 日志记录
使用ROS2日志宏：
```cpp
RCLCPP_INFO(this->get_logger, "常规信息");
RCLCPP_DEBUG(this->get_logger, "调试信息");
RCLCPP_WARN(this->get_logger, "警告信息");
RCLCPP_ERROR(this->get_logger, "错误信息");
RCLCPP_FATAL(this->get_logger, "致命错误");
```

### 配置验证
启动时验证配置参数：
```cpp
/**
 * @brief 配置参数校验模板函数
 * @param value 参数值
 * @param min 最小值（开区间，value 必须 > min）
 * @param max 最大值（闭区间，value 必须 ≤ max）
 * @param name 参数名（用于错误日志）
 * @return true=合法，false=非法
 */
template<typename T>
static bool validateField(T value, T min, T max, const char* name, const rclcpp::Logger& logger) {
    if (value <= min || value > max) {
        RCLCPP_ERROR(logger, "配置错误：%s 超出范围", name);
        return false;
    }
    return true;
}
```

### const使用
- 优先使用`const`修饰不修改的参数
- `const`成员函数用于不修改对象状态的方法

### 智能指针
- 优先使用`unique_ptr`管理独占资源
- 使用`shared_ptr`管理共享资源（如 ROS2 Publisher/Subscriber）
- 避免使用裸指针（除 observer 模式的引用传递外）

## 示例

```cpp
/**
 * @file example.cpp
 * @brief 示例模块 —— 演示项目代码风格
 *
 * 本文件演示了项目中使用的代码风格，包括：
 *   - 文件级 Doxygen 注释
 *   - 类/函数/变量的注释方式
 *   - 分段注释和行内注释
 */

#include <memory>
#include <vector>
#include <string>

// ==================== ROS2核心头文件 ====================
#include "rclcpp/rclcpp.hpp"

// ==================== 本地头文件 ====================
#include "config_manager.h"

// ==================== 常量定义 ====================

/** @brief 最大重试次数（超过后跳过当前路径点） */
static constexpr int MAX_RETRY_COUNT = 3;

// ══════════════════════════════════════════════════════════════
//  数据管理器实现
// ══════════════════════════════════════════════════════════════

/**
 * @brief 机器人运行时数据管理中心 —— 存储自身/敌方血量、位置、云台角度
 *
 * DataManager 是整个系统的"数据仓库"，所有模块通过它读写运行时数据。
 * 线程安全：getSelfHealth 使用 atomic 无锁读取，其他方法使用 mutex 保护。
 */
class DataManager {
    static constexpr int kNumEnemies = 6;  ///< 敌方单位数量

private:
    int player_id_;                                      ///< 玩家ID（1=蓝方，2=红方）
    std::atomic<int> atomic_self_health_;                ///< 当前血量（无锁快速读取）
    std::deque<int> self_health_history_;                ///< 血量历史
    mutable std::mutex mutex_;                           ///< 互斥锁

public:
    /**
     * @brief 构造函数
     * @param player_id 玩家ID（1=蓝方，2=红方）
     */
    explicit DataManager(int player_id) : player_id_(player_id) {
        // 初始化完成
    }

    /**
     * @brief 更新自身血量并计算血量变化量
     * @param health 当前血量
     * @param timestamp 时间戳
     */
    void updateSelfHealth(int health, rclcpp::Time timestamp) {
        std::lock_guard<std::mutex> lock(mutex_);
        atomic_self_health_.store(health, std::memory_order_relaxed);
        self_health_history_.push_back(health);
        // 裁剪过期历史数据...
    }

    /**
     * @brief 获取自身血量（无锁，200Hz 安全）
     * @return 当前血量
     */
    int getSelfHealth const {
        return atomic_self_health_.load(std::memory_order_relaxed);
    }
};
```

## 工具

推荐使用以下工具：
- clang-format：格式化代码
- clang-tidy：静态代码分析

## 更新日志

- 2026-07-20：全面更新为 Doxygen 注释风格
  - 文件级使用 `/** @file @brief */` 格式
  - 函数级使用 `/** @brief @param @return */` 格式
  - 成员变量使用 `///< description` 行尾注释
  - 常量使用 `/** @brief */` 块注释
  - 分段使用 `═══` 分隔符
  - 所有注释使用中文
- 2026-04-15：重写以匹配实际代码风格（使用`//`注释）
- 2026-03-16：初始版本创建（不匹配实际代码）
