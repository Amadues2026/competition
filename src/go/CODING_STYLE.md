# 代码风格规范

本文档描述项目中实际使用的代码风格，基于现有代码总结。

## 基本原则

- 使用4空格缩进，不使用Tab
- 每行尽量控制在120字符以内
- 使用Google风格的大括号（左大括号不换行）
- 指针和引用的`*`和`&`紧靠类型（左侧对齐）

## 命名规范

### 类名
- 使用PascalCase（大驼峰）
- 示例：`TimeManager`, `PathNavigator`, `ShootController`, `DataManager`

### 函数名
- 使用camelCase（小驼峰）
- 示例：`initialize()`, `updateGameTime()`, `getCurrentTime()`, `updateSelfHealth()`

### 变量名
- 使用snake_case（下划线分隔）
- 示例：`game_start_time_`, `current_game_time_`, `player_id_`
- 成员变量以下划线结尾

### 常量名
- 使用UPPER_CASE（全大写+下划线）
- 示例：`MAX_SPEED_FAR`, `CAMERA_FX`, `PATH_COUNT`, `MAP_UNIT_TO_METER`

### 文件名
- 头文件：snake_case.h
- 源文件：snake_case.cpp
- 示例：`config_manager.h`, `path_strategy.cpp`

## 注释风格

### 文件级注释
使用`//`注释，不使用Doxygen风格的`/** */`：
```cpp
// ==================== 文件分隔 ====================
// TensorRT推理库头文件(提供nvinfer1命名空间,用于加载和执行TensorRT推理引擎)
#include <NvInfer.h>

// ==================== ROS2核心头文件 ====================
#include "rclcpp/rclcpp.hpp"
```

### 类注释
使用单行`//`注释，紧跟类定义：
```cpp
class DataManager {  // 定义DataManager类(负责管理机器人的血量、位置、速度等数据,包括自身和敌人的数据)
```

### 成员变量注释
使用`//`行尾注释：
```cpp
int player_id_;  // 玩家ID(1或2,区分己方机器人)
std::atomic<int> atomic_self_health_;  // 原子变量保护当前血量
std::deque<int> self_health_history_;  // 声明成员变量self_health_history_(使用双端队列存储自己血量的历史记录,最多保存5次,用于计算血量变化)
```

### 函数注释
- 简单函数：不添加注释
- 复杂函数：使用`//`注释在函数前
```cpp
// 更新自身血量
void updateSelfHealth(int health, rclcpp::Time timestamp) {
    // ...
}

// 计算血量变化
int getSelfHealthChange() const {
    // ...
}
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
- 函数之间空2行
- 类的成员变量之间空1行
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
1. 第三方库（TensorRT、CUDA等）
2. C标准库
3. C++标准库
4. ROS2头文件
5. OpenCV头文件
6. 本地头文件

使用注释分隔：
```cpp
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
1. 私有成员变量（无访问修饰符，默认private）
2. 公共成员函数
3. 私有成员函数
4. 构造函数

**注意**：与常见规范不同，本项目将成员变量放在最前面。

### 函数组织
- 简短函数（<20行）可以写在类定义中
- 复杂函数（>20行）应在类外实现
- 相关函数放在一起

## 特殊规范

### 原子操作
对于多线程访问的简单类型，使用`std::atomic`：
```cpp
std::atomic<int> atomic_self_health_;  // 原子变量保护当前血量

// 写操作
atomic_self_health_.store(health, std::memory_order_relaxed);

// 读操作
int health = atomic_self_health_.load(std::memory_order_relaxed);
```

### 异常处理
使用`try-catch`捕获异常，使用`RCLCPP_ERROR`记录错误：
```cpp
try {
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize CUDA engine");
    }
} catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "TensorRT初始化失败：%s", e.what());
    throw;  // 重新抛出
}
```

### CUDA错误处理
检查CUDA函数返回值：
```cpp
cudaError_t err = cudaMalloc(&gpu_buffers_[0], input_size_);
if (err != cudaSuccess) {
    RCLCPP_ERROR(this->get_logger(), "CUDA内存分配失败：%s", 
                 cudaGetErrorString(err));
    throw std::runtime_error("CUDA内存分配失败");
}
```

### 日志记录
使用ROS2日志宏：
```cpp
RCLCPP_INFO(this->get_logger(), "常规信息");
RCLCPP_DEBUG(this->get_logger(), "调试信息");
RCLCPP_WARN(this->get_logger(), "警告信息");
RCLCPP_ERROR(this->get_logger(), "错误信息");
RCLCPP_FATAL(this->get_logger(), "致命错误");
```

### 配置验证
启动时验证配置参数：
```cpp
if (max_speed_far_ <= 0 || max_speed_far_ > 100) {
    RCLCPP_WARN(this->get_logger(), 
                "配置验证失败：max_speed_far=%.2f, 使用默认值10.0", 
                max_speed_far_);
    max_speed_far_ = 10.0;
}
```

### 看门狗机制
监控关键循环执行：
```cpp
// 成员变量
std::mutex watchdog_mutex_;
rclcpp::Time control_loop_last_update_;
rclcpp::Time shoot_loop_last_update_;

// 循环开始时更新
{
    std::lock_guard<std::mutex> lock(watchdog_mutex_);
    control_loop_last_update_ = current_time;
}

// 定期检查看门狗
void checkWatchdogs(rclcpp::Time current_time) {
    auto control_elapsed = TimeManager::timeSince(control_loop_last_update_, current_time);
    if (control_elapsed > 0.1) {
        RCLCPP_WARN(this->get_logger(), "控制循环超时！最后更新：%.2f秒前", 
                    control_elapsed);
    }
}
```

### const使用
- 优先使用`const`修饰不修改的参数
- `const`成员函数用于不修改对象状态的方法

### 智能指针
- 优先使用`unique_ptr`管理独占资源
- 使用`shared_ptr`管理共享资源
- 避免使用裸指针

## 示例

```cpp
// ==================== 头文件包含 ====================
#include <memory>
#include <vector>
#include <string>

// ==================== ROS2头文件 ====================
#include "rclcpp/rclcpp.hpp"

// ==================== 本地头文件 ====================
#include "config_manager.h"

// ==================== 常量定义 ====================
const double MAP_UNIT_TO_METER = 0.2;  // 1地图单位 = 0.2m
const int MAX_RETRY_COUNT = 3;  // 最大重试次数

// ==================== 类定义 ====================
class DataManager {  // 定义DataManager类(负责管理机器人的血量、位置、速度等数据)
private:
    int player_id_;  // 玩家ID(1或2,区分己方机器人)
    std::atomic<int> atomic_self_health_;  // 原子变量保护当前血量
    std::deque<int> self_health_history_;  // 血量历史记录
    
public:
    DataManager(int player_id) : player_id_(player_id) {
        // 初始化
    }
    
    void updateSelfHealth(int health, rclcpp::Time timestamp) {
        // 更新血量
    }
    
    int getSelfHealth() const {
        return atomic_self_health_.load(std::memory_order_relaxed);
    }
};
```

## 工具

推荐使用以下工具：
- clang-format：格式化代码
- clang-tidy：静态代码分析

## 注意事项

- 本项目使用`//`注释风格，不使用Doxygen风格的`/** */`
- 类定义使用单行`//`注释
- 成员变量以`_`结尾
- 头文件包含顺序：第三方库 → C++标准库 → ROS2 → OpenCV → 本地头文件
- 使用注释分隔不同类型的头文件

## 更新日志

- 2026-04-15：重写以匹配实际代码风格
  - 使用`//`注释风格
  - 更新头文件包含顺序
  - 更新类和成员变量注释格式
  - 移除Doxygen风格要求
- 2026-03-16：初始版本创建（不匹配实际代码）