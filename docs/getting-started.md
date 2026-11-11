# Getting started

## 1. Obtain the source

```bash
git clone https://github.com/Amadues2026/competition.git
cd competition
```

Commands below assume the repository root. The expected runtime platform is Ubuntu 22.04 with ROS2 Humble. Windows can run the Python dataset tests; the robot application is not a Windows target.

## 2. Install ROS2 and CPU dependencies

Install ROS2 Humble using the [official Ubuntu installation instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html). Then install:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-colcon-common-extensions \
  ros-humble-ros-base ros-humble-ament-cmake-gtest ros-humble-cv-bridge \
  ros-humble-geometry-msgs ros-humble-sensor-msgs ros-humble-std-msgs \
  ros-humble-ament-index-cpp libopencv-dev libyaml-cpp-dev
```

For CPU unit tests only, skip GPU installation and follow [Testing](testing.md).

## 3. Install the GPU toolchain

Install an NVIDIA driver, CUDA toolkit, and TensorRT 10 development libraries appropriate for your GPU and Ubuntu release using the [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) installation guides. Jetson and desktop installations differ; do not assume an x86 package set works on Jetson.

CMake must find `CUDAToolkit`, `NvInfer.h`, and `libnvinfer`. For nonstandard installations, pass `-DCUDAToolkit_ROOT=...`, `-DTENSORRT_INCLUDE_DIR=...`, and `-DTENSORRT_LIBRARY=...` as needed. A version number in this guide is a build target, not evidence that every combination has been validated.

## 4. Build the runtime

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select tdt_interface go --cmake-args \
  -DBUILD_RUNTIME=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

The executable is `S_d2c2` in ROS package `go`. `tdt_interface` supplies the custom messages. If switching from a CPU-only build, explicitly set `BUILD_RUNTIME=ON` again.

## 5. Supply a model and configuration

Use an engine built for the same TensorRT/CUDA/GPU environment. `models/yolov5n_d2c2_fp16.engine` is retained from the original repository; its build environment and accuracy have not been independently documented. Follow [D2C2](../D2C2/README.md) to create a new engine.

Copy the configuration to a deployment-specific location if you need to customize it:

```bash
cp src/go/config/game_config.yaml /tmp/tdt-game-config.yaml
```

For a headless machine, change `detection.enable_visualization` to `false` in that copy. The checked-in camera intrinsics, waypoint coordinates, and unit conversion describe a specific simulator setup. They are not universal robot calibration values. `GoNode::loadConfigParameters()` scales selected speed/position fields; read that function before changing units.

## 6. Start the simulator and application

Start the external simulator separately. The repository provides neither the simulator executable nor a complete simulated world. Confirm its topics with `ros2 topic list`, then run:

```bash
ros2 run go S_d2c2 1 --ros-args \
  -p engine_path:=/absolute/path/to/model.engine \
  -p config_path:=/tmp/tdt-game-config.yaml
```

Use player `2` for the red team. To restrict the configured firing task to one waypoint, the optional positional argument is a one-based index from 1 to 11:

```bash
ros2 run go S_d2c2 1 6 --ros-args -p engine_path:=/absolute/path/to/model.engine
```

The optional index only changes which waypoint has a task; it does not start at that waypoint or supply missing simulator data.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `rclcpp` / ament package not found | Source `/opt/ros/humble/setup.bash` in the current shell. |
| CUDA or TensorRT missing during CPU testing | Configure with `-DBUILD_RUNTIME=OFF`. |
| Runtime exits requesting an engine path | Pass `--ros-args -p engine_path:=...`. |
| Engine deserialization fails | Rebuild the engine on a compatible deployment machine. |
| OpenCV PnP symbols do not link | Include the `calib3d` development component. |
| GUI errors on a server | Disable visualization in the YAML configuration. |
| No position or camera data | Start the external simulator and verify topic names/types. |
| Configuration fails to load | Check the path, required fields, and all 11 waypoints per team. No waypoint fallback is provided. |

For reproducible bug reports include the commit, platform, compiler, ROS distribution, CUDA/TensorRT versions, command, and relevant logs.
