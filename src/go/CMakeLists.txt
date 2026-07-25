cmake_minimum_required(VERSION 3.10)
project(go)

set(CMAKE_CXX_STANDARD 17)

# 使用 ament_auto 自动管理依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tdt_interface REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ament_index_cpp REQUIRED)

# 查找第三方库
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)

# ── 源文件分组 ──
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

# 添加可执行文件
add_executable(S_d2c2
  ${CORE_SOURCES}
  ${UTILITY_SOURCES}
  ${FILTER_SOURCES}
  ${NAVIGATION_SOURCES}
  ${COMBAT_SOURCES}
  ${DETECTION_SOURCES}
)

# 添加ROS2依赖
ament_target_dependencies(S_d2c2
  rclcpp
  geometry_msgs
  tdt_interface
  sensor_msgs
  cv_bridge
  ament_index_cpp
)

# OpenCV 头文件路径（非 target 接口的兼容处理）
target_include_directories(S_d2c2 PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
)

# 链接第三方库
target_link_libraries(S_d2c2
  ${OpenCV_LIBS}
  yaml-cpp
  CUDA::cudart
  curl
  nvinfer
)

# ── 单元测试 ──
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # ── 测试分组 ──

  # waypoint_utils（纯值对象输入，不依赖 YAML 解析器）
  ament_add_gtest(test_waypoint_utils test/test_waypoint_utils.cpp
    src/waypoint_utils.cpp
    src/path_strategy.cpp
  )
  target_include_directories(test_waypoint_utils PRIVATE src)

  # config_manager（验证 YAML 路径点和 EKF 参数解析）
  ament_add_gtest(test_config_manager test/test_config_manager.cpp
    src/config_manager.cpp
    src/rolling_median_filter.cpp
  )
  target_include_directories(test_config_manager PRIVATE src)
  target_compile_definitions(test_config_manager PRIVATE
    TEST_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/game_config.yaml"
  )
  target_link_libraries(test_config_manager yaml-cpp)

  # shoot_controller（验证摆动偏移始终限制在 YAML 默认范围内）
  ament_add_gtest(test_shoot_controller test/test_shoot_controller.cpp
    src/shoot_controller.cpp
  )
  target_include_directories(test_shoot_controller PRIVATE src)
  ament_target_dependencies(test_shoot_controller rclcpp)

  # time_manager（依赖 rclcpp::Time）
  ament_add_gtest(test_time_manager test/test_time_manager.cpp
    src/time_manager.cpp
  )
  target_include_directories(test_time_manager PRIVATE src)
  ament_target_dependencies(test_time_manager rclcpp)

  # data_manager（依赖 TimeManager + rclcpp::Time）
  ament_add_gtest(test_data_manager test/test_data_manager.cpp
    src/data_manager.cpp
    src/time_manager.cpp
  )
  target_include_directories(test_data_manager PRIVATE src)
  ament_target_dependencies(test_data_manager rclcpp)

  # path_navigator（依赖 TimeManager + PathStrategy）
  ament_add_gtest(test_path_navigator test/test_path_navigator.cpp
    src/path_navigator.cpp
    src/path_strategy.cpp
    src/time_manager.cpp
  )
  target_include_directories(test_path_navigator PRIVATE src)
  ament_target_dependencies(test_path_navigator rclcpp)

  # target_ekf（依赖 OpenCV 的 cv::Mat）
  ament_add_gtest(test_target_ekf test/test_target_ekf.cpp
    src/target_ekf.cpp
    src/general_kalman_filter.cpp
  )
  target_include_directories(test_target_ekf PRIVATE src)
  target_link_libraries(test_target_ekf ${OpenCV_LIBS})
endif()

# 安装
install(TARGETS S_d2c2 DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

# ROS2包配置
ament_package()
