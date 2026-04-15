# T-DT 校园战机器人自主控制系统

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![C++](https://img.shields.io/badge/C++-17-blue)
![License](https://img.shields.io/badge/License-Proprietary-red)

## 项目简介

本系统是一个基于 ROS2 的全自主机器人控制系统，专为 T-DT 校园战机器人竞赛设计。系统集成了深度学习视觉识别、智能路径规划、精确射击控制和自适应战术决策，能够在复杂竞技环境中自主完成比赛任务。

## 核心特性

- **全自主运行**：无需人工干预，自动完成整个比赛流程
- **智能视觉识别**：基于 TensorRT GPU 加速的深度学习目标检测（D2C2两分类模型）
- **高性能推理**：推理延迟仅 0.38ms，实时响应能力强
- **多模式射击**：自适应切换自瞄、硬编码、摆动三种射击模式
- **战术决策**：基于血量变化的智能响应和目标优先级管理
- **鲁棒性强**：完善的防卡死、超时重试、异常恢复机制
- **灵活配置**：YAML 配置文件支持，无需重新编译即可调整参数
- **模块化设计**：策略模式、配置管理、性能监控等模块化组件
- **线程安全**：原子操作保护关键数据，避免数据竞争
- **完整日志**：状态转换、关键操作、异常检测的详细日志
- **看门狗机制**：实时监控循环执行，防止卡死
- **配置验证**：启动时自动验证配置参数合理性

## 快速开始

### 环境要求

- Ubuntu 20.04 / 22.04 LTS
- ROS2 Humble Hawksbill
- CMake 3.22+
- GCC/G++ 9.0+
- Python 3.8+
- OpenCV 4.5+
- CUDA Toolkit 12.x
- cuDNN 8.x
- TensorRT 10.x
- yaml-cpp 0.7+

### 安装步骤

```bash
# 1. 激活 ROS2 环境
source /opt/ros/humble/setup.bash

# 2. 编译项目
cd /home/robot/competition
colcon build

# 3. 激活工作空间
source install/setup.bash
```

### 运行程序

```bash
# 运行 Player1（蓝队）
ros2 run go S_d2c2 1

# 运行 Player2（红队）
ros2 run go S_d2c2 2
```

## 项目结构

```
competition/
├── src/
│   └── go/                          # 主程序包
│       ├── src/
│       │   ├── go_d2c2.cpp          # 主程序实现
│       │   ├── config_manager.h/cpp # 配置管理器
│       │   ├── path_strategy.h/cpp  # 路径策略
│       │   ├── performance_monitor.h # 性能监控器
│       │   └── waypoint.h           # 路径点定义
│       ├── config/
│       │   └── game_config.yaml     # 配置文件
│       ├── CODING_STYLE.md          # 代码风格规范
│       ├── CMakeLists.txt           # 构建配置
│       └── package.xml              # 包描述
├── tdt_interface/                   # ROS2 消息接口
├── models/                          # 深度学习模型
│   └── centernet_r18d2c2_fp16.engine
├── D2C2/                            # D2C2 模型项目
│   ├── train/                       # 训练库
│   ├── inference/                   # 推理库
│   ├── datasets/                    # 数据集
│   └── scripts/                     # 脚本工具
└── 说明书.txt                        # 完整技术文档
```

## 核心功能

### 路径导航
- 基于路径点的自动导航系统
- 自适应速度控制（远/中/近距离）
- 防卡死机制（超时检测、自动重试）
- 支持 Player1/Player2 对称路径
- **看门狗监控**：实时检测循环卡死
- **日志记录**：导航过程详细日志

### 视觉检测
- D2C2 两分类模型（红/蓝装甲板）
- TensorRT FP16 加速
- PnP 距离解算
- 运动预测和弹道补偿
- 多级滤波系统
- **异常处理**：模型加载失败检测
- **性能监控**：推理时间统计

### 射击控制
- **自瞄模式**：基于深度学习目标检测
- **硬编码模式**：预设角度射击
- **摆动模式**：自动摆动搜索目标
- 自适应模式切换
- 血量变化监控
- **线程安全**：原子操作保护血量数据
- **栈式管理**：被攻击点位历史记录

### 战术决策
- 基于血量变化的战术响应
- 目标优先级管理
- 被攻击检测
- 位置保持调整
- **配置验证**：启动时参数合理性检查

### 系统可靠性（v6.0新增）
- **异常处理**：
  - TensorRT 初始化异常捕获
  - CUDA 内存分配错误检测
  - 模型文件缺失处理
- **并发安全**：
  - 原子操作保护血量数据
  - 互斥锁保护看门狗数据
- **监控机制**：
  - 看门狗监控控制循环（100ms超时）
  - 看门狗监控射击循环（100ms超时）
  - 性能监控器记录关键指标
- **日志系统**：
  - 状态转换日志（MOVING ↔ SHOOTING）
  - 目标检测日志（检测成功/失败）
  - 射击命令日志（每次射击）
  - 导航移动日志（路径点切换）
  - 异常警告日志（超时、卡死、配置错误）

## 配置说明

主要配置文件：`src/go/config/game_config.yaml`

```yaml
# 玩家配置
player:
  id: 1  # 1=Player1（蓝方）, 2=Player2（红方）

# 速度控制
speed:
  max_far: 10.0       # 远距离速度
  max_near: 3.0       # 近距离速度
  max_precise: 0.8    # 精确模式速度

# 相机参数
camera:
  fx: 554.26   # 焦距 X
  fy: 554.26   # 焦距 Y
  cx: 320.0    # 光心 X
  cy: 240.0    # 光心 Y

# 弹道参数
ballistics:
  bullet_speed: 23.0  # 子弹速度 (m/s)
  gravity: 5.0        # 重力加速度

# 摆动参数
swing:
  range: 6.0   # 摆动范围 (±6度)
  speed: 10.0  # 摆动速度 (度/秒)

# 阈值参数（新增）
threshold:
  consecutive_miss_disable: 10  # 连续未命中禁用自瞄
  consecutive_miss_swing: 3     # 连续未命中切换摆动
  hit: -10                      # 击中阈值

# 超时参数（新增）
timeout:
  move: 10.0              # 移动超时（秒）
  health_history: 5.0     # 血量历史时长（秒）
  other_data: 10.0        # 其他数据时长（秒）

# 滤波参数（新增）
filter:
  distance_window: 5      # 距离滤波窗口
  yaw_alpha: 0.1          # Yaw滤波系数
  pitch_alpha: 0.1        # Pitch滤波系数
```

完整配置说明请参考 [说明书.txt](./说明书.txt)

## D2C2 模型训练

```bash
# 进入训练目录
cd /home/robot/competition/D2C2/train

# 安装依赖
pip install -r requirements_train.txt

# 开始训练
python3 train_2class.py

# 转换为 TensorRT 引擎
cd ../scripts
bash convert_to_tensorrt.sh

# 复制到模型目录
cp ../inference/centernet_r18d2c2_fp16.engine ../../models/
```

详细训练说明请参考 [D2C2/README.md](./D2C2/README.md)

## 性能指标

- **推理延迟**：0.38ms（TensorRT FP16）
- **检测速度**：约 2348 QPS
- **模型大小**：2.6MB
- **控制循环**：200Hz（5ms 定时器）
- **射击循环**：33.3Hz（30ms 定时器）
- **摆动循环**：50Hz（20ms 定时器）
- **位置调整**：20Hz（50ms 定时器）
- **健康检查**：1Hz（1000ms 定时器）

## ROS2 接口

### 订阅话题
- `/position_player_X` - 机器人位置（100Hz）
- `/real_angles_player_X` - 云台实时角度（100Hz）
- `/camera_image_player_X` - 相机图像（30Hz）
- `/game_healths` - 血量数据（10Hz）
- `/game_time` - 游戏时间（1Hz）

### 发布话题
- `/target_speed_player_X` - 目标速度（200Hz，5ms定时器）
- `/target_angles_player_X` - 目标角度 + 射击指令（50Hz，20ms定时器）

## 开发指南

### 代码风格
项目遵循 [CODING_STYLE.md](./src/go/CODING_STYLE.md) 规范

### 格式化代码
```bash
cd /home/robot/competition/src/go
find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

### 修改配置参数
1. 编辑 `src/go/config/game_config.yaml`
2. 重新编译（可选）：`colcon build --packages-select go`
3. 运行程序：`ros2 run go S_d2c2 1`
4. 检查日志确认配置验证通过

### 添加新功能
1. 在 `game_config.yaml` 中添加配置
2. 在 `config_manager.h/cpp` 中添加解析代码
3. 在 `go_d2c2.cpp` 的 `validateConfig()` 中添加验证
4. 在 `go_d2c2.cpp` 中实现功能
5. 添加适当的日志记录
6. 编译测试

## 日志说明（v6.0新增）

系统提供多级日志记录：

### INFO 级别
- 状态转换：`P1: 状态切换 MOVING -> SHOOTING`
- 目标检测：`P1: 检测到目标 center=(320.0,240.0) score=0.85`
- 射击命令：`P1: 发布射击命令`
- 导航移动：`P1: 移动到路径点 P2`

### DEBUG 级别
- 详细的调试信息（开发时使用）

### WARN 级别
- 配置参数警告：`配置验证失败：max_speed_far=0.0，使用默认值10.0`
- 看门狗警告：`控制循环超时！最后更新: 0.15秒前`

### ERROR 级别
- 异常错误：`TensorRT初始化失败：模型文件不存在`
- CUDA错误：`CUDA内存分配失败：out of memory`

## 故障排查

### 程序无法启动
```bash
# 检查 ROS2 环境
echo $ROS_DISTRO

# 检查模型文件
ls /home/robot/competition/models/centernet_r18d2c2_fp16.engine

# 检查配置文件
ls /home/robot/competition/src/go/config/game_config.yaml

# 查看启动日志
ros2 run go S_d2c2 1 2>&1 | grep -E "ERROR|WARN"
```

### 视觉检测失败
```bash
# 检查模型文件
ls -lh /home/robot/competition/models/centernet_r18d2c2_fp16.engine

# 如果模型不存在，重新训练
cd /home/robot/competition/D2C2/train
python3 train_2class.py
```

### 机器人不移动
```bash
# 检查话题发布
ros2 topic echo /target_speed_player_1

# 检查节点状态
ros2 node list
ros2 node info /go_node

# 检查配置验证
ros2 run go S_d2c2 1 2>&1 | grep "配置"
```

### 看门狗超时警告
```bash
# 查看看门狗日志
ros2 run go S_d2c2 1 2>&1 | grep "超时"

# 可能原因：
# 1. 控制循环卡死（检查死锁）
# 2. 系统负载过高（检查CPU使用率）
# 3. 定时器配置错误（检查代码）
```

更多故障排查信息请参考 [说明书.txt](./说明书.txt) 第七部分

## 相关文档

- [说明书.txt](./说明书.txt) - 完整技术文档（3130行，含v6.0新增功能）
- [操作手册.pdf](./操作手册.pdf) - 仿真器操作说明
- [规则手册.doc](./规则手册.doc) - 比赛规则
- [D2C2/README.md](./D2C2/README.md) - D2C2 模型说明
- [CODING_STYLE.md](./src/go/CODING_STYLE.md) - 代码风格规范

## 版本历史

- **v6.0** (2026-04-15) - 系统可靠性增强
  - 添加 TensorRT 初始化异常处理
  - 添加 CUDA 内存分配异常处理
  - 使用原子操作保护血量数据
  - 栈式管理被攻击状态
  - 添加状态转换日志
  - 添加关键操作日志
  - 添加配置验证函数
  - 添加看门狗机制
- **v5.0** (2026-03-16) - 添加配置管理、路径策略、性能监控
- **v4.2** (2026-03-11) - 添加 D2C2 两分类模型
- **v4.0** (2026-02-22) - 升级为 TensorRT 深度学习检测
- **v3.0** (2026-02-12) - 完整重写技术文档

## 技术支持

如有问题，请：
1. 查看 [说明书.txt](./说明书.txt) 故障排查章节
2. 检查日志输出（ERROR/WARN级别）
3. 检查配置文件是否正确
4. 检查 ROS2 官方文档
5. 查看代码注释

## 许可声明

本系统仅用于 T-DT 校园战机器人竞赛，未经授权不得用于其他用途。

---

**开发团队** | **版本** 6.0 | **更新日期** 2026-04-15