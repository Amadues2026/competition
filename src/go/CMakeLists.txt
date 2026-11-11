cmake_minimum_required(VERSION 3.18)
project(go)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(BUILD_RUNTIME "Build the ROS2 GPU runtime (disable for CPU-only tests)" ON)

# Find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
if(BUILD_RUNTIME)
  find_package(geometry_msgs REQUIRED)
  find_package(tdt_interface REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(ament_index_cpp REQUIRED)
endif()

# Find third-party libraries
if(BUILD_RUNTIME)
  find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs highgui calib3d)
  find_package(CUDAToolkit REQUIRED)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h REQUIRED)
  find_library(TENSORRT_LIBRARY nvinfer REQUIRED)
else()
  find_package(OpenCV 4 REQUIRED COMPONENTS core)
endif()
find_package(yaml-cpp REQUIRED)

# Source groups
set(CORE_SOURCES
  src/go_d2c2.cpp
  src/go_node.cpp
  src/logger.cpp
)

set(UTILITY_SOURCES
  src/time_manager.cpp
  src/data_manager.cpp
  src/performance_monitor.cpp
  src/config_manager.cpp
  src/waypoint_utils.cpp
  src/path_strategy.cpp
)

set(FILTER_SOURCES
  src/general_kalman_filter.cpp
  src/rolling_median_filter.cpp
  src/target_ekf.cpp
)

set(NAVIGATION_SOURCES
  src/path_navigator.cpp
  src/movement_controller.cpp
)

set(COMBAT_SOURCES
  src/tracking_state_machine.cpp
  src/shoot_controller.cpp
  src/combat_manager.cpp
  src/state_manager.cpp
)

set(DETECTION_SOURCES
  src/yolov5_detector.cpp
  src/visualizer.cpp
)

if(BUILD_RUNTIME)
# Build the GPU runtime only when requested.
add_executable(S_d2c2
  ${CORE_SOURCES}
  ${UTILITY_SOURCES}
  ${FILTER_SOURCES}
  ${NAVIGATION_SOURCES}
  ${COMBAT_SOURCES}
  ${DETECTION_SOURCES}
)

# ROS2 target dependencies
ament_target_dependencies(S_d2c2
  rclcpp
  std_msgs
  geometry_msgs
  tdt_interface
  sensor_msgs
  cv_bridge
  ament_index_cpp
)

# Include directories for OpenCV and SDK headers
target_include_directories(S_d2c2 PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${cv_bridge_INCLUDE_DIRS}
)

# Link third-party libraries
target_link_libraries(S_d2c2
  ${OpenCV_LIBS}
  yaml-cpp
  CUDA::cudart
  ${TENSORRT_LIBRARY}
)

endif()

# Unit tests do not require a GPU or a TensorRT engine.
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # Test groups

  # Waypoint value objects and index helpers
  ament_add_gtest(test_waypoint_utils test/test_waypoint_utils.cpp
    src/waypoint_utils.cpp
    src/path_strategy.cpp
  )
  target_include_directories(test_waypoint_utils PRIVATE src)

  # YAML waypoint and EKF configuration parsing
  ament_add_gtest(test_config_manager test/test_config_manager.cpp
    src/config_manager.cpp
    src/rolling_median_filter.cpp
  )
  target_include_directories(test_config_manager PRIVATE src)
  target_compile_definitions(test_config_manager PRIVATE
    TEST_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/game_config.yaml"
  )
  target_link_libraries(test_config_manager yaml-cpp)

  # Task-controller sweep bounds
  ament_add_gtest(test_shoot_controller test/test_shoot_controller.cpp
    src/shoot_controller.cpp
  )
  target_include_directories(test_shoot_controller PRIVATE src)
  ament_target_dependencies(test_shoot_controller rclcpp)

  # TimeManager uses rclcpp::Time
  ament_add_gtest(test_time_manager test/test_time_manager.cpp
    src/time_manager.cpp
  )
  target_include_directories(test_time_manager PRIVATE src)
  ament_target_dependencies(test_time_manager rclcpp)

  # DataManager uses TimeManager and rclcpp::Time
  ament_add_gtest(test_data_manager test/test_data_manager.cpp
    src/data_manager.cpp
    src/time_manager.cpp
  )
  target_include_directories(test_data_manager PRIVATE src)
  ament_target_dependencies(test_data_manager rclcpp)

  # PathNavigator uses TimeManager and PathStrategy
  ament_add_gtest(test_path_navigator test/test_path_navigator.cpp
    src/path_navigator.cpp
    src/path_strategy.cpp
    src/time_manager.cpp
  )
  target_include_directories(test_path_navigator PRIVATE src)
  ament_target_dependencies(test_path_navigator rclcpp)

  # TargetEKF uses OpenCV matrices
  ament_add_gtest(test_target_ekf test/test_target_ekf.cpp
    src/target_ekf.cpp
    src/general_kalman_filter.cpp
  )
  target_include_directories(test_target_ekf PRIVATE src)
  target_include_directories(test_target_ekf PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(test_target_ekf ${OpenCV_LIBS})
endif()

# Installation
if(BUILD_RUNTIME)
  install(TARGETS S_d2c2 DESTINATION lib/${PROJECT_NAME})
endif()
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

# ROS2 package metadata
ament_package()
