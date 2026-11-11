# Testing

## Python dataset tools

Python 3.10+ is sufficient; training dependencies are not needed:

```bash
python -m unittest discover -s tests -v
python D2C2/train/train_yolov5.py --help
bash -n D2C2/scripts/convert_to_tensorrt.sh
```

Tests cover normalized labels, both class mappings, empty labels for background images, disjoint/reproducible splits, nested paths, invalid geometry and paths, missing images, and CLI use from another directory. Temporary fixtures use the system temporary directory.

## C++ unit tests on Ubuntu 22.04 / ROS2 Humble

```bash
sudo apt-get install -y ros-humble-ros-base ros-humble-ament-cmake-gtest \
  python3-colcon-common-extensions libopencv-dev libyaml-cpp-dev
source /opt/ros/humble/setup.bash
colcon build --packages-select tdt_interface go --cmake-args \
  -DBUILD_RUNTIME=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
source install/setup.bash
colcon test --packages-select go --return-code-on-test-failure
colcon test-result --all --verbose
```

`BUILD_RUNTIME=OFF` skips the GPU executable, CUDA, TensorRT, and GUI-related discovery. It does not remove the ROS2 dependency from modules that use `rclcpp::Time` or logging.

| Executable | Main coverage |
| --- | --- |
| `test_waypoint_utils` | Health indices, waypoint mapping, thresholds, task lookup, path strategy. |
| `test_config_manager` | YAML waypoint/parameter loading and median-window validation. |
| `test_shoot_controller` | Simulator task state and bounded sweep offsets. |
| `test_time_manager` | Time initialization, elapsed game time, stop/reset behavior. |
| `test_data_manager` | Health/angle history and time windows. |
| `test_path_navigator` | Sequencing, return/retry behavior, terminal conditions. |
| `test_target_ekf` | Filter updates, resets, convergence cases, innovation statistics. |

The authoritative count and result are the test runner's output, not a fixed README number.

## CI scope

`.github/workflows/ci.yml` runs Python tests on 3.10 and 3.12 and the C++ unit tests on ROS2 Humble. It reports failures using `--return-code-on-test-failure` and retains C++ result artifacts for seven days. It does not compile or execute the TensorRT runtime in the CPU job.

A passing CPU workflow does not validate model accuracy, TensorRT engine compatibility, simulator integration, callback latency, or operation on a physical robot. Those checks require the appropriate environment. There is no current reproducible benchmark suite for those claims.

## GPU integration checks

Before documenting a successful runtime deployment, record the commit, GPU, driver, CUDA, TensorRT, exported tensor contract, engine generation command, simulator version, and exact launch command. Build with `BUILD_RUNTIME=ON`, test engine loading, inspect topic traffic, and compare detections with a held-out dataset. Do not reuse training images as validation evidence.
