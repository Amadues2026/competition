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

# 查找第三方库
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

# 添加头文件路径
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  /usr/local/cuda-13.1/include
)

# 添加库文件路径
link_directories(
  /usr/local/cuda-13.1/lib64
)

# 添加可执行文件
add_executable(S_d2c2
  src/go_d2c2.cpp
  src/config_manager.cpp
  src/path_strategy.cpp
)

# 添加ROS2依赖
ament_target_dependencies(S_d2c2
  rclcpp
  geometry_msgs
  tdt_interface
  sensor_msgs
  cv_bridge
)

# 链接第三方库
target_link_libraries(S_d2c2
  ${OpenCV_LIBS}
  yaml-cpp
  cudart
  curl
  nvinfer
)

# 安装
install(TARGETS S_d2c2 DESTINATION lib/${PROJECT_NAME})

# ROS2包配置
ament_package()