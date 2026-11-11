# Learning guide

This is a practical reading path through the current repository. Start with tests; the simulator and GPU runtime can come later.

## 1. Understand the boundary

Read the README and [architecture guide](architecture.md). The application receives data from an external simulator and publishes commands back. It is not a complete simulator, SLAM stack, or general-purpose robotics framework.

## 2. Run the smallest useful check

Run the Python dataset tests from [Testing](testing.md). Read `prepare_dataset.py` alongside `tests/test_prepare_dataset.py`. Check how COCO corner coordinates become normalized center coordinates and why training and validation images must be disjoint.

## 3. Learn the value objects and indexing

Read `waypoint.h`, `path_strategy.h/cpp`, `waypoint_utils.h/cpp`, then `test_waypoint_utils.cpp`. Follow one waypoint from YAML coordinates to an index in the visitation sequence. Notice the distinction between one-based waypoint labels and zero-based indices.

## 4. Follow configuration

Read `game_config.yaml`, `GameConfig`, `ConfigManager::loadFromFile()`, and `GoNode::loadConfigParameters()`. Trace a camera intrinsic, a speed value, and a waypoint. Identify which values are converted to map units and which are not. Inspect `validateConfig()` before changing parameter ranges.

## 5. Follow time and history

Read `TimeManager`, `DataManager`, and their tests. Track how a timestamp enters a history window and expires. These modules use ROS2 time types, so their C++ tests still need ROS2 even without a GPU.

## 6. Trace node assembly

Read `go_d2c2.cpp` and the `GoNode` constructor. Match each subscriber and publisher to the topic table in the architecture guide. Draw the two callback groups and note where locks protect shared state.

## 7. Read controllers through tests

Read `PathNavigator` and its tests before `MovementController`. Then read the task controller tests and state transitions. Keep simulator-specific rules separate from reusable data structures when proposing changes.

## 8. Inspect perception last

Read `detection_types.h`, `YOLOv5Detector`, `TargetEKF`, and `Visualizer`. Verify the model tensor contract and preprocessing before trying a new checkpoint. Run the filter tests without treating them as evidence of detector accuracy.

## 9. Make a reproducible contribution

Choose a specific issue, describe its observed behavior, make a focused change, and run the relevant tests. For a runtime problem, include the simulator and GPU environment. For a documentation change, verify commands and links. See [Contributing](../CONTRIBUTING.md).

## Original references

The large original learning notes and technical manual are preserved in [reference](reference/README.md), including historical examples that may predate the current layout. Use current source code and these setup/test guides when commands conflict.
