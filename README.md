![CI](https://github.com/Amadues2026/competition/actions/workflows/ci.yml/badge.svg)
![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![C++](https://img.shields.io/badge/C++-17-blue)
![TensorRT](https://img.shields.io/badge/TensorRT-FP16-green)
![YOLOv5n](https://img.shields.io/badge/YOLOv5n-nano-yellow)
![EKF](https://img.shields.io/badge/EKF-3--state-blue)
![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04|22.04-orange)

# T-DT 校园战 · 机器人全自主控制系统

> ROS2 + YOLOv5n + TensorRT + 3-state EKF — 全自主视觉识别、路径导航、弹道射击、战术决策

---

## 特性

| 模块 | 技术方案 | 性能指标 |
|------|----------|----------|
| 视觉检测 | YOLOv5n + TensorRT FP16 | ~2ms/帧, mAP50 0.90 |
| 目标追踪 | 3-state EKF (yaw/pitch/dist 随机游走) | 同济方案，自适应 R/Q |
| 目标选择 | 距离门限 (30m 上限 + 10m 跳变) + 最近优先 | 过滤远距离/跨目标干扰 |
| 弹道补偿 | 重力补偿 + 飞行时间修正 | 23m/s 子弹速度 |
| 射击控制 | 自瞄 → 硬编码 → 摆动搜索 (三级降级) | 见到就打, 自动恢复 |
| 运动制动 | 检测到目标自动减速 | 翻倍减速阈值 |
| 配置管理 | YAML 全参数可配（含蓝红双方路径点） | 调点无需重新编译 |
| 可视化 | 独立 GUI 工作线程 | 默认约 15FPS，不阻塞相机回调 |
| 架构分解 | 20 源文件 + 7 测试, 3 Controller + SharedContext + RuntimeConfig | 解耦可测试 |

## 系统架构

```
                          ┌─────────────────┐
                          │   Camera Input   │
                          │    640x480 BGR   │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │   YOLOv5n D2C2  │
                          │  TensorRT FP16  │
                          │  224x416 → 6xN  │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  敌我过滤 + 距离闸门 │
                          │  30m 上限 / 10m 跳变 │
                          │  选择最近有效目标  │
                          └────────┬────────┘
                                   │
                           ┌────────▼────────┐
                           │ 3-state EKF     │   CombatManager
                           │ yaw/pitch/dist   │   (30ms shootLoop)
                           │ 随机游走 + 自适应R│
                          └────────┬────────┘
                                   │
                     ┌──────────────┼──────────────┐
                     │              │              │
            ┌────────▼────────┐    │    ┌─────────▼────────┐
            │  PnP 距离解算    │    │    │   弹道补偿         │
            └────────┬────────┘    │    └─────────┬────────┘
                     │              │              │
                     └──────────────┼──────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  SharedContext      │ ← 数据交换中心
                          │  state, current_yaw │   (mutex 保护)
                          │  target_locked, ... │
                          └──┬──────────┬───────┘
                             │          │
              ┌──────────────┼──────────┼──────────────┐
              │              │          │              │
      ┌────────▼──────┐  ┌───▼──────┐  │  ┌───────────▼─────┐
      │ MovementCtrl  │  │ StateMgr │  │  │ GoNode          │
      │ 5ms control   │  │ 1s health│  │  │ MultiThreaded   │
      │ 1s position   │  │ Check    │  │  │ Executor(2)     │
      └───────────────┘  └──────────┘  │  │ 5 sub + 2 pub   │
                                       │  └──────┬──────────┘
                                      │
                             ┌────────▼─────────┐
                             │  swingLoop 20ms   │ ← CombatManager
                             │  摆动 + 射击指令   │
                             └──────────────────┘
```

## 快速开始

### 环境要求

- Ubuntu 22.04 / ROS2 Humble
- CUDA 12+ / TensorRT 10+
- OpenCV 4.x / yaml-cpp

### 编译运行

```bash
cd /home/robot/competition
source /opt/ros/humble/setup.bash
colcon build --packages-select tdt_interface go
source install/setup.bash

# 蓝方
ros2 run go S_d2c2 1

# 红方
ros2 run go S_d2c2 2
```

## 运行测试

```bash
colcon build --packages-select go --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select go
```

7 个测试套件，47 个测试项，覆盖纯逻辑模块（无需 ROS2 环境或硬件）。

## CI/CD

推送到 `main` 或创建 PR 时自动触发：

```bash
# 构建 + 47 个测试项（GitHub Actions, ubuntu-22.04）
# 需安装：ROS2 Humble, OpenCV, yaml-cpp, CUDA Toolkit, TensorRT
```

详情见 [ci.yml](.github/workflows/ci.yml)。

## 模型训练

```bash
conda activate RM
pip install -r D2C2/train/requirements_train.txt
cd D2C2/train
python3 train_yolov5.py
```

训练流程：COCO 标注 → YOLO 格式转换 → 下载 yolov5n.pt → 训练 100 epoch (Early Stopping) → ONNX 导出 → TensorRT FP16 转换 → 自动部署到 models/

## 射击模式

| 模式 | 触发条件 | 原理 | 降级条件 |
|------|----------|------|----------|
| 自瞄 | 进入 SHOOTING 状态 | 3-state EKF + PnP + 弹道补偿 | 连续 10 帧无有效检测 |
| 硬编码 | 连续 10 次血量变化未达标 | 航点预设 yaw/pitch 角度 | 连续 3 次血量变化未达标 |
| 摆动搜索 | 硬编码再连续 3 次未达标 | ±6 度, 10 度/s 摆动搜索 | 检测到血量变化后停止 |

## 项目结构

```
src/go/
├── .clang-format              # 代码格式化配置
├── CMakeLists.txt             # 构建配置 (20 源文件 + 7 测试)
├── CODING_STYLE.md            # 编码规范
├── package.xml                # ROS2 包描述
├── config/
│   └── game_config.yaml       # 全参数 YAML 配置
└── src/
    ├── go_d2c2.cpp            # 可执行入口 (main → GoNode)
    ├── go_node.h/cpp          # ROS2 节点薄胶水 (Visualizer 集成, 无 test_waypoints_)
    ├── shared_context.h       # WithMutex<T> 封装 chassis/gimbal, atomic 检测状态 (95 行)
    ├── runtime_config.h       # 只读配置参数 (120 行)
    ├── movement_controller.h/cpp  # 导航 + 调速 + 超时 + 漂移
    ├── combat_manager.h/cpp   # YOLO + 3-state EKF (自适应 Q/R) + 弹道 + 摆动
    ├── state_manager.h/cpp    # 血量评估 + 降级 + 回撤
    ├── tracking_state_machine.h/cpp # LOST→DETECTING→TRACKING→TEMP_LOST
    ├── waypoint_utils.h/cpp   # 比赛规则常量 + 纯函数
    ├── data_manager.h/cpp     # 时间窗口数据管理
    ├── shoot_controller.h/cpp # 射击状态机 (自瞄/硬编码/摆动)
    ├── path_navigator.h/cpp   # 路径遍历 + 重试
    ├── config_manager.h/cpp   # 公开强类型配置；YAML 解析细节仅在 cpp
    ├── path_strategy.h/cpp    # 仅生成固定访问索引，不持有路径点
    ├── performance_monitor.h/cpp  # 命名计时器性能统计
    ├── logger.h/cpp           # 日志过滤器
    ├── general_kalman_filter.h/cpp # 通用 KF 底层
    ├── target_ekf.h/cpp       # 3-state EKF (yaw/pitch/dist, 自适应 Q/R)
    ├── rolling_median_filter.h/cpp  # 滚动中值滤波
    ├── time_manager.h/cpp     # 游戏时间管理
    ├── yolov5_detector.h/cpp  # TensorRT YOLOv5 推理
    ├── detection_types.h      # 检测结果轻量共享类型（不依赖 TensorRT）
    ├── visualizer.h/cpp       # 异步 GUI 线程（与 ROS 回调解耦）
    ├── waypoint.h             # Waypoint 结构体
    └── test/
        ├── test_waypoint_utils.cpp     # 7 个测试
        ├── test_time_manager.cpp      # 4 个测试
        ├── test_data_manager.cpp      # 6 个测试
        ├── test_path_navigator.cpp    # 12 个测试
        ├── test_target_ekf.cpp        # 7 个测试
        ├── test_config_manager.cpp    # YAML 参数 + 非法滤波窗口
        └── test_shoot_controller.cpp  # 有限范围摆动偏移

D2C2/
├── train/train_yolov5.py      # 训练+导出脚本
├── datasets/RMCOCO2/          # COCO 数据集
├── scripts/                   # TensorRT 转换
└── README.md

models/
└── yolov5n_d2c2_fp16.engine   # TensorRT FP16 引擎

tdt_interface/msg/
├── SendData.msg               # 角度+射击指令
└── ReceiveData.msg            # 实时角度回传
```

## 技术栈

| 层 | 组件 | 说明 |
|----|------|------|
| 通信 | ROS2 Humble | Publisher/Subscriber |
| 语言 | C++17 | 27 个 .cpp（20 源文件 + 7 测试）+ 23 个 .h/.hpp |
| 检测 | YOLOv5n (ultralytics) | nano backbone, ~2.5M 参数 |
| 推理 | TensorRT 10.x FP16 | enqueueV3 |
| 追踪 | 3-state EKF | yaw/pitch/dist 随机游走 + 自适应 R/Q |
| 配置 | yaml-cpp | game_config.yaml |
| 加速 | CUDA 12+ | GPU 推理 |

## 文档

- [学习路径.md](学习路径.md) — 新队员全项目复现完整手册
- [说明书.txt](说明书.txt) — 架构设计 / 类参考 / 算法细节 / 故障排查
- [.github/workflows/ci.yml](.github/workflows/ci.yml) — CI/CD 自动构建与测试
- [CODING_STYLE.md](src/go/CODING_STYLE.md) — 命名规范 / 注释风格
- [D2C2/README.md](D2C2/README.md) — 数据集格式 / 训练参数
