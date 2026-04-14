# 代码风格规范

本文档定义了项目中C++代码的统一风格规范。

## 基本原则

- 使用4空格缩进，不使用Tab
- 每行最大120字符
- 使用Google风格的大括号（左大括号不换行）
- 指针和引用的*和&紧靠类型（左侧对齐）

## 命名规范

### 类名
- 使用PascalCase（大驼峰）
- 示例：`TimeManager`, `PathNavigator`, `ShootController`

### 函数名
- 使用camelCase（小驼峰）
- 示例：`initialize()`, `updateGameTime()`, `getCurrentTime()`

### 变量名
- 使用snake_case（下划线分隔）
- 示例：`game_start_time_`, `current_game_time_`, `player_id_`
- 成员变量以下划线结尾

### 常量名
- 使用UPPER_CASE（全大写+下划线）
- 示例：`MAX_SPEED_FAR`, `CAMERA_FX`, `PATH_COUNT`

### 文件名
- 头文件：snake_case.h
- 源文件：snake_case.cpp
- 示例：`config_manager.h`, `path_strategy.cpp`

## 注释规范

### 文件头注释
每个文件应包含文件头注释：
```cpp
/**
 * @file filename.h
 * @brief 简短描述
 * @details 详细描述
 */
```

### 类注释
每个类应包含Doxygen风格的注释：
```cpp
/**
 * @brief 类的简短描述
 * @details 类的详细描述
 */
class ClassName {
    // ...
};
```

### 函数注释
公共函数应包含参数和返回值说明：
```cpp
/**
 * @brief 函数简短描述
 * @param param1 参数1说明
 * @param param2 参数2说明
 * @return 返回值说明
 */
ReturnType functionName(Type1 param1, Type2 param2);
```

### 成员变量注释
每个成员变量应有行尾注释：
```cpp
int player_id_;  ///< 玩家ID（1或2）
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
星号和&紧靠类型（左侧对齐）：
```cpp
int* ptr = nullptr;
const std::string& ref = str;
```

## 头文件包含顺序

1. 对应的头文件（如果有）
2. C标准库
3. C++标准库
4. 第三方库（ROS2、OpenCV等）
5. 本地头文件

使用注释分隔：
```cpp
// C标准库
#include <cmath>

// C++标准库
#include <vector>
#include <string>

// ROS2
#include "rclcpp/rclcpp.hpp"

// OpenCV
#include <opencv2/opencv.hpp>

// 本地头文件
#include "config_manager.h"
```

## 代码组织

### 类成员顺序
1. 公共类型定义（typedef、enum）
2. 静态常量
3. 构造函数和析构函数
4. 公共成员函数
5. 保护成员
6. 私有成员函数
7. 私有成员变量

### 函数组织
- 简短函数（<20行）可以写在类定义中
- 复杂函数（>20行）应在类外实现
- 相关函数放在一起

## 其他规范

### const使用
- 优先使用const修饰不修改的参数
- const成员函数用于不修改对象状态的方法
- 使用constexpr定义编译时常量

### 智能指针
- 优先使用unique_ptr管理独占资源
- 使用shared_ptr管理共享资源
- 避免使用裸指针

### 异常处理
- 使用try-catch捕获可能抛出异常的代码
- 使用RCLCPP_ERROR记录错误
- 提供有意义的错误信息

### 日志
- 使用RCLCPP_INFO记录常规信息
- 使用RCLCPP_DEBUG记录调试信息
- 使用RCLCPP_WARNING记录警告
- 使用RCLCPP_ERROR记录错误
- 使用RCLCPP_FATAL记录致命错误

## 示例

```cpp
/**
 * @file example.h
 * @brief 示例头文件
 * @details 演示代码风格规范
 */

#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <string>
#include <vector>

// C++标准库
#include <memory>

/**
 * @brief 示例类
 * @details 演示类的组织结构
 */
class ExampleClass {
public:
    /**
     * @brief 构造函数
     * @param name 名称
     */
    explicit ExampleClass(const std::string& name);

    /**
     * @brief 获取名称
     * @return 名称
     */
    std::string getName() const;

    /**
     * @brief 设置名称
     * @param name 新名称
     */
    void setName(const std::string& name);

private:
    std::string name_;  ///< 名称
    int count_;         ///< 计数
};

#endif  // EXAMPLE_H
```

## 工具

推荐使用以下工具自动检查和修复代码风格：
- clang-format：格式化代码
- clang-tidy：静态代码分析
- cpplint：Google代码风格检查器

## 违规处理

代码审查时应检查以下违规：
- 缺少注释
- 命名不规范
- 格式不符合规范
- 未使用const修饰
- 魔法数字（应定义为常量）

## 更新日志

- 2026-03-16：初始版本创建