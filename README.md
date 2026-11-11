# T-DT Robotics Competition Controller

[![CI](https://github.com/Amadues2026/competition/actions/workflows/ci.yml/badge.svg)](https://github.com/Amadues2026/competition/actions/workflows/ci.yml)

A C++17 / ROS2 Humble controller for the T-DT campus robotics competition simulator. The project combines camera-based detection, target tracking, waypoint navigation, and competition-specific task management in a modular codebase.

The repository is maintained by [Amadues2026](https://github.com/Amadues2026). It is intended as a practical reference for student robotics teams. It is an early-stage project; no broad adoption or independently reproduced benchmark is claimed.

## What is included

- ROS2 integration for simulator position, camera, angle feedback, health, and game time.
- TensorRT inference for a two-class red/blue armor detector.
- Tracking, filtering, waypoint sequencing, configuration loading, and simulator task controllers.
- YAML configuration with no duplicated hard-coded waypoint table.
- Eight C++ test executables and a standard-library Python dataset test suite.
- Dataset conversion, disjoint train/validation lists, training, and explicit model export tools.

## Start here

| Goal | Guide |
| --- | --- |
| Install dependencies, build, and run | [Getting started](docs/getting-started.md) |
| Understand modules and simulator topics | [Architecture](docs/architecture.md) |
| Learn the code in a useful order | [Learning guide](docs/learning-guide.md) |
| Prepare data, train, or export a model | [D2C2 model pipeline](D2C2/README.md) |
| Run tests and understand CI scope | [Testing](docs/testing.md) |
| Report a bug or contribute a change | [Contributing](CONTRIBUTING.md) |

## Quick start: tests without a GPU

The Python dataset tools require Python 3.10 or later and no third-party packages:

```bash
python -m unittest discover -s tests -v
python D2C2/train/train_yolov5.py --help
```

C++ unit tests require Ubuntu 22.04, ROS2 Humble, OpenCV, yaml-cpp, and GoogleTest via ament, but do not require CUDA, TensorRT, a model, or the simulator:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select tdt_interface go --cmake-args -DBUILD_RUNTIME=OFF -DBUILD_TESTING=ON
source install/setup.bash
colcon test --packages-select go --return-code-on-test-failure
colcon test-result --all --verbose
```

Use a clean workspace when switching from test-only to runtime builds; `BUILD_RUNTIME` is cached by CMake.

## Runtime requirements

- Ubuntu 22.04 and ROS2 Humble; C++17 and CMake 3.18+.
- OpenCV 4 with `core`, `imgproc`, `imgcodecs`, `highgui`, and `calib3d`; yaml-cpp.
- A compatible NVIDIA GPU, driver, CUDA toolkit, and TensorRT 10 installation.
- The external T-DT simulator or matching publishers; the simulator is not included.
- A compatible TensorRT engine. Rebuild engines for the deployment environment rather than assuming the checked-in binary is portable.

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select tdt_interface go --cmake-args -DBUILD_RUNTIME=ON -DBUILD_TESTING=ON
source install/setup.bash
ros2 run go S_d2c2 1 --ros-args -p engine_path:=/absolute/path/to/model.engine
```

Player `1` selects blue; player `2` selects red. Runtime startup now requires an explicit `engine_path`; the former machine-specific `/home/robot/competition` default is no longer used. See the setup guide for configuration and headless operation.

## Repository map

```text
src/go/                 ROS2 application, configuration, and C++ unit tests
tdt_interface/          Simulator command and feedback messages
D2C2/train/             Dataset preparation, training, and ONNX export
D2C2/scripts/           TensorRT engine conversion
models/                 Existing deployment engine (environment-specific)
tests/                  Python regression tests
docs/                   English setup, architecture, testing, and learning guides
docs/reference/         Preserved original technical and competition documents
.github/                CI and contribution templates
```

## Validation and limitations

CI checks CPU unit tests and dataset tooling. It does not demonstrate GPU inference, model accuracy, simulator integration, or hardware performance. Earlier documentation quoted approximately 2 ms inference and mAP50 0.90, but no reproducible evaluation artifacts accompany those numbers, so they are not presented as verified benchmarks here.

The dataset is not distributed. The old training script used the same image directory for training and validation; new runs use disjoint image lists. For correlated video frames, split by recording/session as well before reporting generalization results. The new export path uses Ultralytics YOLOv5u; validate it against the runtime's documented tensor contract before deployment.

## License and provenance

The ROS package manifests declare Apache-2.0; [LICENSE](LICENSE) supplies that license text for repository-authored software. Third-party libraries, pretrained weights, datasets, the existing engine, and competition documents retain their own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the current provenance gaps and dependencies.
