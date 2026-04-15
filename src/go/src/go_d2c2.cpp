// ==================== TensorRT头文件 ====================
#include <NvInfer.h>  // 包含TensorRT推理库头文件(提供nvinfer1命名空间,用于加载和执行TensorRT推理引擎)
#include <cuda_runtime_api.h>  // 包含CUDA运行时API头文件(提供cudaMalloc,cudaMemcpy等CUDA函数,用于GPU内存管理)
#include <fstream>  // 包含C++文件流库(提供std::ifstream,用于从磁盘文件读取TensorRT引擎数据)
#include <iostream>  // 包含C++标准输入输出库(提供std::cout和std::cerr,用于向控制台打印调试信息和错误信息)
#include <algorithm>  // 包含C++算法库(提供sort,max,min,find等STL算法,用于数据排序和查找)

#include <cmath>  // 包含C++数学库(提供sin,cos,sqrt,abs,atan2等数学函数,用于坐标计算和角度转换)
#include <vector>  // 包含C++标准库vector头文件(提供动态数组容器std::vector,用于存储可变长度的数据集合)
#include <memory>  // 包含C++智能指针库(提供std::unique_ptr和std::shared_ptr,用于自动管理TensorRT对象的内存生命周期)
#include <string>  // 包含C++标准库string头文件(提供std::string类,用于字符串操作)
#include <tuple>  // 包含C++标准库tuple头文件(提供元组std::tuple,用于函数返回多个值或将不同类型的数据组合在一起)
#include <deque>  // 包含C++双端队列库(提供std::deque容器,用于存储历史数据并支持高效的头部和尾部插入删除操作)

// ==================== ROS2核心头文件 ====================
#include "rclcpp/rclcpp.hpp"  // 包含ROS2 C++客户端库头文件(提供rclcpp::Node等ROS2核心类)

// ==================== ROS2消息类型头文件 ====================
#include "geometry_msgs/msg/twist_stamped.hpp"  // 包含ROS2速度消息头文件(提供geometry_msgs::msg::TwistStamped类型)
#include "geometry_msgs/msg/pose_stamped.hpp"  // 包含ROS2位姿消息头文件(提供geometry_msgs::msg::PoseStamped类型)
#include "tdt_interface/msg/send_data.hpp"  // 包含自定义发送数据消息头文件(提供tdt_interface::msg::SendData类型)
#include "tdt_interface/msg/receive_data.hpp"  // 包含自定义接收数据消息头文件(提供tdt_interface::msg::ReceiveData类型)
#include "std_msgs/msg/int32_multi_array.hpp"  // 包含ROS2整数数组消息头文件(提供std_msgs::msg::Int32MultiArray类型)
#include "std_msgs/msg/int32.hpp"  // 包含ROS2整数消息头文件(提供std_msgs::msg::Int32类型)
#include "sensor_msgs/msg/image.hpp"  // 包含ROS2图像消息头文件(提供sensor_msgs::msg::Image类型)
#include "cv_bridge/cv_bridge.h"  // 包含cv_bridge头文件(提供ROS2图像和OpenCV图像之间的转换)

// ==================== OpenCV头文件 ====================
#include <opencv2/opencv.hpp>  // 包含OpenCV库头文件(提供图像处理功能如resize,cvtColor,solvePnP等,用于预处理相机图像和PnP定位)
#include <opencv2/imgproc.hpp>  // 包含OpenCV图像处理头文件(提供resize,cvtColor等图像处理函数)
#include <opencv2/calib3d.hpp>  // 包含OpenCV相机标定头文件(提供solvePnP等相机标定函数)

// ==================== 性能监控头文件 ====================
#include "performance_monitor.h"  // 包含性能监控器头文件(提供PerformanceMonitor类,用于记录推理延迟、FPS等性能指标)
#include "config_manager.h"  // 包含配置管理器头文件(提供ConfigManager类,用于从YAML文件加载配置)
#include "path_strategy.h"  // 包含路径策略头文件(提供IPathPlanningStrategy接口和策略实现)
#include <unordered_map>  // 包含C++无序映射头文件(提供std::unordered_map容器)
#include <stack>  // 包含栈头文件(提供std::stack容器)

// ==================== 路径点定义头文件 ====================
#include "waypoint.h"  // 包含Waypoint结构体定义（统一在waypoint.h中定义）


// ==================== 常量定义 ====================
// 速度单位换算：1地图单位 = 0.2m，因此m/s × 5 = 地图单位/s
const double MAP_UNIT_TO_METER = 0.2;  // 1地图单位 = 0.2m
const double METER_TO_MAP_UNIT = 5.0;  // 1m = 5地图单位

// ==================== 数据保存时长常量 ====================
const double HEALTH_HISTORY_TOLERANCE = 0.1;  // 血量历史时间容忍度（秒），用于判断是否有足够的历史数据
const int MAX_REASONABLE_HEALTH_CHANGE = 500;  // 最大合理血量变化值（HP），超过此值视为异常数据
const int MAX_HEALTH_CHANGES = 10;  // 最大保存的血量变化记录数

// ==================== 射击控制常量 ====================
const int MAX_RETRY_COUNT = 3;  // 最大重试次数，机器人卡住后最多重试3次

// ==================== 通用常量 ====================
const std::tuple<double, double> ZERO_TUPLE = {0.0, 0.0};  // 零值元组（用于默认返回值）
const size_t PATH_COUNT = 11;  // 路径点总数（P1-P11）

// ==================== 装甲板参数 ====================
const double ARMOR_WIDTH = 1.0;  // 装甲板宽度（地图单位，200mm ÷ 0.2）
const double ARMOR_HEIGHT = 0.625;  // 装甲板高度（地图单位，125mm ÷ 0.2）

// ==================== 血量变化标准 ====================
const int HEALTH_CHANGE_THRESHOLD_P4 = -50;  // P4路径点血量变化阈值(-50HP表示血量减少50HP)
const int HEALTH_CHANGE_THRESHOLD_P5 = 0;  // P5路径点血量变化阈值(0表示没有变化)
const int HEALTH_CHANGE_THRESHOLD_P6 = -60;  // P6路径点血量变化阈值(-60HP表示血量减少60HP)
const int HEALTH_CHANGE_THRESHOLD_BASE = -120;  // 基地血量变化阈值(-120HP表示血量减少120HP)

// ==================== 路径点配置 ====================
// Player1路径点（蓝方）
const std::vector<Waypoint> WAYPOINTS_PLAYER1 = {  // 玩家1的路径点数组(从基地到P4到P5到P6到敌方基地)
    {15.9, -44.9, 0.0, 0.0, false},  // 玩家1的P4路径点(敌方前哨站,x=15.9,y=-44.9,无射击任务)
    {5.8, -44.9, 0.0, 0.0, false},  // 玩家1的P5路径点(补给站,x=5.8,y=-44.9,无射击任务)
    {5.8, -37.8, 0.0, 0.0, false},  // 玩家1的P6路径点(进攻点,x=5.8,y=-37.8,无射击任务)
    {-27.7, -38.3, 243.0, -1.5, true},  // P4 - 攻击5号敌人
    {-27.2, -2.4, 89.5, 2.5, true},  // P5 - 攻击4号敌人
    {-2.5, -2.8, 58.7, 19.5, true},  // P6 - 攻击前哨站
    {-3.0, 32.0, 112.8, 8.0, true},  // 玩家1的敌方基地路径点(需要射击任务,x=-3.0,y=32.0)
    {6.8, 32.0, 48.5, 8.0, true},  // 玩家1的敌方基地路径点(需要射击任务,x=6.8,y=32.0)
    {6.8, 47.5, 314.0, 8.0, true},  // 玩家1的敌方基地路径点2(需要射击任务,x=6.8,y=47.5)
    {-6.0, 47.5, 221.0, 8.0, true},  // P10 - 攻击基地
    {-6.0, 32.0, 123.0, 8.0, true}  // P11 - 攻击基地
};

// Player2路径点（红方，坐标中心对称，yaw/pitch保持与Player1相同）
const std::vector<Waypoint> WAYPOINTS_PLAYER2 = {  // 玩家2的路径点数组(从基地到P4到P5到P6到敌方基地)
    {-15.9, 44.9, 0.0, 0.0, false},  // 玩家2的P4路径点(敌方前哨站,x=-15.9,y=44.9)
    {-5.8, 44.9, 0.0, 0.0, false},  // 玩家2的P5路径点(补给站,x=-5.8,y=44.9)
    {-5.8, 37.8, 0.0, 0.0, false},  // 玩家2的P6路径点(进攻点,x=-5.8,y=37.8)
    {27.7, 38.3, 243.0, -1.5, true},  // P4 - 坐标取反，yaw/pitch同Player1
    {27.2, 2.4, 89.5, 2.5, true},  // P5 - 坐标取反，yaw/pitch同Player1
    {2.5, 2.8, 58.7, 19.5, true},  // P6 - 坐标取反，yaw/pitch同Player1
    {3.0, -32.0, 112.8, 8.0, true},  // P7 - 坐标取反，yaw/pitch同Player1
    {-6.8, -32.0, 48.5, 8.0, true},  // P8 - 坐标取反，yaw/pitch同Player1
    {-6.8, -47.5, 314.0, 8.0, true},  // P9 - 坐标取反，yaw/pitch同Player1
    {6.0, -47.5, 221.0, 8.0, true},  // P10 - 坐标取反，yaw/pitch同Player1
    {6.0, -32.0, 123.0, 8.0, true}  // P11 - 坐标取反，yaw/pitch同Player1
};

// ==================== 时间管理类 ====================
class TimeManager {  // 定义TimeManager类(负责管理游戏时间,包括游戏开始时间、当前游戏时长、初始化状态等)
private:
    static inline rclcpp::Time game_start_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);  // 静态成员变量game_start_time_(存储游戏开始时间,初始化为ROS时间0)
    static inline int32_t current_game_time_ = 0;  // 静态成员变量current_game_time_(存储当前游戏时长,单位:秒,从游戏开始到现在经过的时间,用于判断比赛阶段)
    static inline bool initialized_ = false;  // 静态成员变量initialized_(标志变量,true表示时间管理器已经初始化完成,可以正常使用)

public:
    // 初始化时间管理器，设置游戏开始时间（只能调用一次，重复调用将被忽略）
    static void initialize(rclcpp::Time ros_time) {
        if (!initialized_) {
            game_start_time_ = ros_time;
            current_game_time_ = 0;
            initialized_ = true;
        }
    }

    // 更新当前游戏时长（每帧调用，用于判断比赛阶段）
    static void updateGameTime(int32_t game_time) {
        current_game_time_ = game_time;
    }

    // 获取当前游戏时间
    static rclcpp::Time getCurrentTime() {
        return game_start_time_ + rclcpp::Duration::from_seconds(current_game_time_);
    }

    // 计算两个时间点之间的时间差（用于判断数据是否过期）
    static double timeSince(rclcpp::Time timestamp, rclcpp::Time current_time) {
        return (current_time - timestamp).seconds();
    }

    // 判断是否超过指定时间（超时检测）
    static bool isTimeout(rclcpp::Time start_time, rclcpp::Time current_time, double timeout_seconds) {
        return timeSince(start_time, current_time) > timeout_seconds;
    }

    // 统计过期记录数量（假设timestamps是有序的，遇到未过期记录时停止遍历）
    static size_t countExpiredRecords(rclcpp::Time current_time,
                                       const std::deque<rclcpp::Time>& timestamps,
                                       double max_duration) {
        if (timestamps.size() <= 1) return 0;

        size_t expire_count = 0;
        for (const auto& ts : timestamps) {
            if (timeSince(ts, current_time) > max_duration) {
                expire_count++;
            } else {
                break;
            }
        }

        return expire_count;
    }
};  // TimeManager类定义结束

// ==================== 滤波器类 ====================
class MovingAverageFilter {
private:
    std::deque<double> values_;  // 声明双端队列values_(存储最近N个数值用于计算平均值)
    int window_size_;  // 声明整数变量window_size_(存储滑动窗口大小)
public:
    MovingAverageFilter(int window_size = 5) : window_size_(window_size) {}  // 声明构造函数MovingAverageFilter(初始化移动平均滤波器,设置窗口大小)
    
    void update(double value) {  // 声明函数update(更新滤波器,添加新值)
        values_.push_back(value);  // 将新值添加到队列末尾
        while (values_.size() > window_size_) {  // 循环删除直到队列长度不超过窗口大小
            values_.pop_front();  // 移除队列最前面的值(保持窗口大小不变)
        }
    }
    
    double getFiltered() const {  // 声明函数getFiltered(获取滤波后的值)
        if (values_.empty()) return 0.0;  // 如果队列为空,返回0
        double sum = 0.0;  // 声明浮点数变量sum(用于累加队列中的所有值)
        for (auto v : values_) sum += v;  // 遍历队列,累加所有值
        return sum / values_.size();  // 返回平均值(滤波后的值)
    }
    
    void reset() {  // 声明函数reset(重置滤波器,清空队列)
        values_.clear();  // 清空队列(删除所有历史值)
    }
};

class ExponentialSmoothingFilter {  // 指数平滑滤波器类(用于平滑角度变化)
private:
    double filtered_value_;  // 声明浮点数变量filtered_value_(存储滤波后的值)
    double alpha_;  // 声明浮点数变量alpha_(存储平滑系数,范围0-1,值越大响应越快)
    bool initialized_;  // 声明布尔变量initialized_(标记滤波器是否已初始化)
public:
    ExponentialSmoothingFilter(double alpha = 0.3) : filtered_value_(0.0), alpha_(alpha), initialized_(false) {}  // 声明构造函数ExponentialSmoothingFilter(初始化指数平滑滤波器,设置平滑系数)
    
    double update(double new_value) {  // 声明函数update(更新滤波器,计算新的滤波值)
        if (!initialized_) {  // 如果滤波器未初始化,执行对应操作
            filtered_value_ = new_value;  // 将滤波值直接设为新值(首次更新)
            initialized_ = true;  // 标记滤波器已初始化
        } else {  // 否则(滤波器已初始化)
            filtered_value_ = alpha_ * new_value + (1.0 - alpha_) * filtered_value_;  // 计算新的滤波值(指数平滑公式:新值*alpha+旧值*(1-alpha))
        }
        return filtered_value_;  // 返回滤波后的值
    }
    
    void reset() {  // 声明函数reset(重置滤波器)
        initialized_ = false;  // 标记滤波器未初始化
        filtered_value_ = 0.0;  // 重置滤波值为0
    }
};

// ==================== 数据管理类 ====================
class DataManager {  // 定义DataManager类(负责管理机器人的血量、位置、速度等数据,包括自身和敌人的数据)
private:
    int player_id_;  // 玩家ID(1或2,区分己方机器人)
    
    // 自身血量数据（使用原子操作保护，避免数据竞争）
    std::atomic<int> atomic_self_health_;  // 原子变量保护当前血量
    std::deque<int> self_health_history_;  // 声明成员变量self_health_history_(使用双端队列存储自己血量的历史记录,最多保存5次,用于计算血量变化)
    std::deque<rclcpp::Time> self_health_timestamps_;  // 声明成员变量self_health_timestamps_(使用双端队列存储自己血量数据的时间戳,用于判断数据是否超过5秒需要清理)
    int initial_self_health_;  // 声明成员变量initial_self_health_(存储自己的初始血量,用于计算血量变化量,单位:HP)
    bool initial_self_health_set_;  // 声明成员变量initial_self_health_set_(标志变量,true表示初始血量已经设置,防止重复设置)
    
    // 自身血量变化（最新10组）
    std::deque<int> self_health_changes_;  // 声明成员变量self_health_changes_(使用双端队列存储血量变化量的历史记录,最多保存10组,用于分析攻击模式)
    
    // 敌方6个单位血量数据
    std::deque<int> enemy_health_history_[6];  // 声明成员数组enemy_health_history_(6个双端队列,每个队列存储一个敌方机器人的血量历史记录)
    std::deque<rclcpp::Time> enemy_health_timestamps_[6];  // 声明成员数组enemy_health_timestamps_(6个双端队列,每个队列存储一个敌方机器人的血量时间戳)
    int initial_enemy_health_[6];  // 声明成员数组initial_enemy_health_(6个整数,存储6个敌方机器人的初始血量,用于计算击杀进度)
    bool initial_enemy_health_set_[6];  // 声明成员数组initial_enemy_health_set_(6个布尔值,标记6个敌方机器人的初始血量是否已设置)
    
    // 敌方血量变化（最新10组）
    std::deque<int> enemy_health_changes_[6];  // 声明成员数组enemy_health_changes_(6个双端队列,每个队列存储一个敌方机器人的血量变化量历史)
    
    // 位置数据（10秒内）
    std::deque<std::tuple<double, double, rclcpp::Time>> position_history_;  // 声明成员变量position_history_(使用双端队列存储位置历史记录,每个元素包含x坐标、y坐标和时间戳,最多保存10秒的数据)
    
    // 速度数据（10秒内）
    std::deque<std::tuple<double, double, rclcpp::Time>> velocity_history_;  // 声明成员变量velocity_history_(使用双端队列存储速度历史记录,每个元素包含vx速度、vy速度和时间戳,最多保存10秒的数据)
    
    // Yaw/Pitch数据（10秒内）
    std::deque<std::tuple<double, double, rclcpp::Time>> yaw_pitch_history_;  // 声明成员变量yaw_pitch_history_(使用双端队列存储云台角度历史记录,每个元素包含偏航角、俯仰角和时间戳,最多保存10秒的数据)
    
    // 配置参数
    double health_history_duration_;
    double other_data_duration_;

public:
        DataManager(int player_id, double health_history_duration = 5.0, double other_data_duration = 10.0) 
                : player_id_(player_id), 
                  atomic_self_health_(0),
                  initial_self_health_(0), 
                  initial_self_health_set_(false),
                  health_history_duration_(health_history_duration),
                  other_data_duration_(other_data_duration) {
                for (int i = 0; i < 6; i++) {  // 循环6次,初始化所有6个敌方机器人的数据
                        initial_enemy_health_[i] = 0;  // 初始化敌方血量(默认值,后续会更新)
                        initial_enemy_health_set_[i] = false;  // 标记敌方血量未初始化(等待首次检测)
        }
    }

      // 处理自身血量变化(用于判断是否被攻击)
        void updateSelfHealth(int health, rclcpp::Time timestamp) {
                if (!initial_self_health_set_) {  // 如果初始血量未设置,先设置初始值
                        initial_self_health_ = health;  // 将initial_self_health_赋值为health
                        initial_self_health_set_ = true;  // 将initial_self_health_set_设置为true
        }

                // 使用原子操作更新当前血量（避免数据竞争）
                atomic_self_health_.store(health, std::memory_order_relaxed);

                self_health_history_.push_back(health);
                self_health_timestamps_.push_back(timestamp);  // 将时间戳添加到历史记录末尾

        // 保留health_history_duration_秒内的数据
                size_t expire_count = TimeManager::countExpiredRecords(
                        timestamp, self_health_timestamps_, health_history_duration_);  // 传入健康历史记录时长
        
                for (size_t i = 0; i < expire_count; i++) {  // 循环expire_count次,删除所有过期的记录
                        self_health_history_.pop_front();
                        self_health_timestamps_.pop_front();  // 删除最旧的时间戳记录
        }

          // 通过对比health_history_duration_秒前和当前的血量判断是否受伤
        // 只有当有足够的历史数据（至少health_history_duration_秒的数据）时才计算
                if (self_health_history_.size() <= 1) {
                        return;
                }

              // 确保有足够的历史数据来计算血量变化
                        if (self_health_timestamps_.empty()) {  // 如果血量历史记录为空,直接返回
                                return;
                        }

                                double time_to_oldest = TimeManager::timeSince(self_health_timestamps_.front(), timestamp);  // 计算时间差(用于判断数据是否过期)
                
                // 只有当最早的数据至少有health_history_duration_秒时才计算血量变化
                                if (time_to_oldest < health_history_duration_ - HEALTH_HISTORY_TOLERANCE) {  // 如果没有足够的历史数据,直接返回
                                        return;
                                }

                                        int health_5s_ago = self_health_history_.front();
                                        int change = health - health_5s_ago;  // 计算血量变化
                    
                    // 只有当变化合理时才保存（避免初始数据错误导致的异常值）
                    // 假设单次血量变化不会超过500
                                        if (std::abs(change) > MAX_REASONABLE_HEALTH_CHANGE) {  // 如果血量变化不合理,直接返回
                                                return;
                                        }

                                                self_health_changes_.push_back(change);  // 记录血量变化(最多保存10组用于分析)
                                                while (self_health_changes_.size() > MAX_HEALTH_CHANGES) {  // 循环删除直到队列长度不超过最大数量
                                                        self_health_changes_.pop_front();  // 移除最旧的血量变化记录(只保留最近10组)
                                                }
    }

      // 处理敌方血量变化(用于判断击杀进度)
        void updateEnemyHealth(int enemy_index, int health, rclcpp::Time timestamp) {
                if (enemy_index < 0 || enemy_index >= 6) return;  // 将>赋值为6) return

                if (!initial_enemy_health_set_[enemy_index]) {  // 如果该敌人的初始血量还未设置,先设置初始值
                        initial_enemy_health_[enemy_index] = health;
                        initial_enemy_health_set_[enemy_index] = true;  // 设置initial_enemy_health_set_[enemy_index]为true
        }

                enemy_health_history_[enemy_index].push_back(health);
                enemy_health_timestamps_[enemy_index].push_back(timestamp);  // 将敌方血量时间戳添加到历史记录末尾

        // 保留health_history_duration_秒内的数据
                size_t expire_count = TimeManager::countExpiredRecords(
                        timestamp, enemy_health_timestamps_[enemy_index], health_history_duration_);
        
                for (size_t i = 0; i < expire_count; i++) {
                        enemy_health_history_[enemy_index].pop_front();
                        enemy_health_timestamps_[enemy_index].pop_front();  // 删除敌方最旧的时间戳记录
        }

          // 通过对比health_history_duration_秒前和当前的血量判断是否受伤
        // 只有当有足够的历史数据（至少health_history_duration_秒的数据）时才计算
                if (enemy_health_history_[enemy_index].size() <= 1) {
                        return;
                }

              // 确保有足够的历史数据来计算血量变化
                        if (enemy_health_timestamps_[enemy_index].empty()) {  // 如果该敌人的血量时间戳队列为空,直接返回
                                return;
                        }

                                double time_to_oldest = TimeManager::timeSince(enemy_health_timestamps_[enemy_index].front(), timestamp);
                
                // 只有当最早的数据至少有health_history_duration_秒时才计算血量变化
                                if (time_to_oldest < health_history_duration_ - HEALTH_HISTORY_TOLERANCE) {
                                        return;
                                }

                                        int health_5s_ago = enemy_health_history_[enemy_index].front();
                                        int change = health - health_5s_ago;
                    
                    // 只有当变化合理时才保存（避免初始数据错误导致的异常值）
                    // 假设单次血量变化不会超过500
                                        if (std::abs(change) > MAX_REASONABLE_HEALTH_CHANGE) {
                                                return;
                                        }

                                                enemy_health_changes_[enemy_index].push_back(change);  // 将敌方血量变化量添加到变化记录末尾
                                                while (enemy_health_changes_[enemy_index].size() > MAX_HEALTH_CHANGES) {  // 循环删除直到队列长度不超过最大数量
                                                        enemy_health_changes_[enemy_index].pop_front();  // 删除敌方最旧的变化记录(保持队列长度)
                                                }
    }

      // 更新机器人位置(用于路径导航和卡住检测)
        void updatePosition(double x, double y, rclcpp::Time timestamp) {
                position_history_.push_back({x, y, timestamp});
        
        // 保留other_data_duration_秒内的数据
                size_t expire_count = TimeManager::countExpiredRecords(
                        timestamp, self_health_timestamps_, other_data_duration_);  // 传入其他数据历史时长
        
                for (size_t i = 0; i < expire_count; i++) {
                        position_history_.pop_front();
        }
    }

    // 更新Yaw/Pitch
        void updateYawPitch(double yaw, double pitch, rclcpp::Time timestamp) {
                yaw_pitch_history_.push_back({yaw, pitch, timestamp});
        
        // 保留other_data_duration_秒内的数据
                size_t expire_count = TimeManager::countExpiredRecords(
                        timestamp, self_health_timestamps_, other_data_duration_);
        
                for (size_t i = 0; i < expire_count; i++) {
                        yaw_pitch_history_.pop_front();
        }
    }

    // 获取当前自身血量（使用原子操作，避免数据竞争）
        int getSelfHealth() const {
                return atomic_self_health_.load(std::memory_order_relaxed);
    }

    // 获取自身血量变化
        int getSelfHealthChange() const {  // 获取自身血量变化量函数
                if (self_health_changes_.empty()) return 0;  // 返回0表示没有结果或操作失败
                return self_health_changes_.back();  // 获取return self_health_changes_的最后一个元素
    }

    // 获取敌方血量变化
        int getEnemyHealthChange(int enemy_index) const {  // 获取敌方血量变化量函数
                if (enemy_index < 0 || enemy_index >= 6) return 0;
                if (enemy_health_changes_[enemy_index].empty()) return 0;
                return enemy_health_changes_[enemy_index].back();  // 获取return enemy_health_changes_[enemy_index]的最后一个元素
    }

    // 获取初始血量
        int getInitialEnemyHealth(int enemy_index) const {  // 获取敌方初始血量函数
                if (enemy_index < 0 || enemy_index >= 6) return 0;
                return initial_enemy_health_[enemy_index];  // 返回指定敌人的初始血量
    }

    // 检查敌方血量是否已初始化
        bool isEnemyHealthInitialized(int enemy_index) const {  // 检查敌方血量是否已初始化函数
                if (enemy_index < 0 || enemy_index >= 6) return false;  // 将>赋值为6) return false
                return initial_enemy_health_set_[enemy_index];  // 返回指定敌人血量是否已初始化
    }
        int getCurrentEnemyHealth(int enemy_index) const {  // 获取当前敌方血量函数
                if (enemy_index < 0 || enemy_index >= 6) return 0;
                if (enemy_health_history_[enemy_index].empty()) return 0;
                return enemy_health_history_[enemy_index].back();
    }

    // 获取前一秒的Yaw/Pitch
        std::tuple<double, double> getHistoricalYawPitch(rclcpp::Time current_time) {
                for (auto it = yaw_pitch_history_.rbegin(); it != yaw_pitch_history_.rend(); ++it) {
                        if (TimeManager::timeSince(std::get<2>(*it), current_time) >= 1.0) {
                                return {std::get<0>(*it), std::get<1>(*it)};  // 返回结构体
            }
        }
                if (!yaw_pitch_history_.empty()) {
                        return {std::get<0>(yaw_pitch_history_.back()), std::get<1>(yaw_pitch_history_.back())};
        }
                return ZERO_TUPLE;  // 没有历史数据时返回零值(表示无法获取角度)
    }

    // 获取当前位置
        std::tuple<double, double> getCurrentPosition() const {  // 返回机器人的当前位置(x,y坐标)
                if (position_history_.empty()) return ZERO_TUPLE;
                return {std::get<0>(position_history_.back()), std::get<1>(position_history_.back())};
    }
};

// ==================== 路径导航类 ====================
class PathNavigator {  // 定义PathNavigator类(负责管理机器人的移动路径,包括路径点导航、卡住检测、路径切换等)
private:
    std::vector<size_t> path_indices_;  // 声明成员变量path_indices_(使用动态数组存储当前路径的路径点索引顺序,用于导航)
    size_t current_target_index_;  // 声明成员变量current_target_index_(存储当前要去第几个路径点的索引,用于导航)
    int player_id_;  // 声明成员变量player_id_(存储玩家ID,1表示玩家1,2表示玩家2,用于区分不同的机器人和路径)
    int stuck_retry_count_;  // 声明成员变量stuck_retry_count_(存储机器人卡住后重试的次数,超过3次就跳过当前路径点)
    rclcpp::Time start_time_to_current_;  // 声明成员变量start_time_to_current_(记录开始移动到当前路径点的时间,用于判断是否超时)
    bool is_returning_;  // 是否正在返回上一位置
    size_t return_to_index_;  // 声明成员变量return_to_index_(存储返回目标的路径点索引,用于返回之前的路径点)
    bool max_retries_exceeded_;  // 声明成员变量max_retries_exceeded_(标志变量,true表示超过最大重试次数,需要跳过当前路径点)
    bool is_in_p7_p11_loop_;  // 声明成员变量is_in_p7_p11_loop_(标志变量,true表示进入了P7-P11循环模式)
    static const int MAX_RETRY_COUNT = 3;  // 定义静态常量MAX_RETRY_COUNT(最大重试次数,设置为3,机器人卡住后最多重试3次)
    ConfigManager* config_manager_;  // 配置管理器指针
    double move_timeout_;  // 移动超时时间
    
    // 路径策略（策略模式）
    std::unique_ptr<IPathPlanningStrategy> path_strategy_;  // 路径规划策略指针

public:
        PathNavigator(int player_id, ConfigManager* config_manager) : player_id_(player_id), current_target_index_(0),  // 声明构造函数PathNavigator(初始化路径导航器,传入玩家ID,加载对应的路径点)
                                                                    stuck_retry_count_(0), is_returning_(false),  // 初始化成员变量(重试计数器设为0,返回标志设为false)
                                                                    return_to_index_(0), max_retries_exceeded_(false),  // 是否已经重试太多次了
                                                                    is_in_p7_p11_loop_(false), config_manager_(config_manager), move_timeout_(10.0) {
        // 使用策略模式创建路径策略
        path_strategy_ = PathStrategyFactory::createStrategy(player_id_);
        
        // 使用策略生成路径索引
        path_indices_ = path_strategy_->generatePathIndices();
        
        // 如果策略创建失败，使用默认的顺序路径
        if (path_indices_.empty()) {
                for (size_t i = 0; i < PATH_COUNT; i++) {
                        path_indices_.push_back(i);
                }
        }
        
        // 从ConfigManager读取move_timeout
        if (config_manager && config_manager->isLoaded()) {
                const auto& config = config_manager->getConfig();
                move_timeout_ = config.timeout.move;
        }
    }

        size_t getCurrentWaypointIndex() const {  // 声明成员函数getCurrentWaypointIndex(获取当前要去第几个路径点,返回索引值)
                if (current_target_index_ < path_indices_.size()) {
                        return path_indices_[current_target_index_];
        }
                return 0;
    }

    // 移动到下一点
        void moveToNext(rclcpp::Time current_time) {
                if (is_returning_) {  // 如果条件满足,执行对应操作
                        current_target_index_ = return_to_index_;
                        is_returning_ = false;  // 将is_returning_设置为false
            return_to_index_ = 0;  // 将return_to_index_初始化为0
                        RCLCPP_DEBUG(rclcpp::get_logger("PathNavigator"), "返回到索引%zu", current_target_index_);
                        return;  // 返回函数(退出函数,不返回值)
        }

                current_target_index_++;
        
        // 检查是否需要进入P7-P11循环
                if (current_target_index_ >= path_indices_.size()) {
                        if (!is_in_p7_p11_loop_) {  // 如果条件满足,执行对应操作
                                path_indices_.clear();  // 清空数组
                                for (size_t i = 6; i < 11; i++) {  // 将i赋值为6; i < 11; i++) {
                                        path_indices_.push_back(i);
                }
                                current_target_index_ = 0;
                                is_in_p7_p11_loop_ = true;  // 将is_in_p7_p11_loop_设置为true
                        } else {  // 否则(如果不满足if条件,执行else分支)
                                current_target_index_ = 0;
            }
        }

                stuck_retry_count_ = 0;  // 机器人开始移动时重置(表示机器人不再卡住)
                max_retries_exceeded_ = false;  // 机器人重新移动后清除该标志
                start_time_to_current_ = current_time;  // 将start_time_to_current_赋值为current_time
    }

    // 返回上一点位
        void returnToPrevious() {  // 声明成员函数returnToPrevious(返回到上一个路径点,用于P7-P11循环)
                if (current_target_index_ > 0) {
                        return_to_index_ = current_target_index_ - 1;
                        is_returning_ = true;  // 将is_returning_设置为true
        }
    }

    // 取消返回状态
        void cancelReturn() {
                is_returning_ = false;
                return_to_index_ = 0;
                RCLCPP_DEBUG(rclcpp::get_logger("PathNavigator"), "取消返回状态");  // 输出取消返回状态的日志(调试信息)
    }

    // 跳过当前点位
        void skipCurrentPoint(rclcpp::Time current_time) {
                stuck_retry_count_ = 0;
                max_retries_exceeded_ = false;
                moveToNext(current_time);  // 切换到下一个路径点(推进比赛进度)
    }

      // 判断是否超过指定时间(用于超时检测)
        bool isTimeout(rclcpp::Time current_time) {
                if (start_time_to_current_.nanoseconds() == 0) {  // 如果条件满足,执行对应操作
                        start_time_to_current_ = current_time;
        }
                return TimeManager::isTimeout(start_time_to_current_, current_time, move_timeout_);  // 判断是否超时(用于卡住检测)
    }

    // 增加重试计数
        void incrementRetryCount(rclcpp::Time current_time) {
                stuck_retry_count_++;  // 重试计数器加1(记录机器人卡住后重试的次数)
                if (stuck_retry_count_ >= MAX_RETRY_COUNT) {  // 如果重试次数达到最大值(超过3次,需要跳过当前路径点)
                        max_retries_exceeded_ = true;  // 重试次数过多时标记为卡住(需要切换路径点)
        }
                start_time_to_current_ = current_time;
    }

    // 重置超时计时
        void resetTimeoutTimer(rclcpp::Time current_time) {
                start_time_to_current_ = current_time;
    }

    // 重置重试计数
        void resetRetryCount() {
                stuck_retry_count_ = 0;
                max_retries_exceeded_ = false;
    }

        int getRetryCount() const { return stuck_retry_count_; }  // 返回结果
        bool isMaxRetriesExceeded() const { return max_retries_exceeded_; }
        bool isReturning() const { return is_returning_; }  // 返回结果
};

// ==================== 射击控制器类 ====================
class ShootController {  // 射击控制器类(管理射击逻辑)
private:
    bool auto_aim_enabled_;  // 声明成员变量auto_aim_enabled_(自动瞄准启用标志,true表示已启用)
    bool auto_aim_disabled_;  // 声明成员变量auto_aim_disabled_(自动瞄准禁用标志,true表示被禁用)
    bool is_shooting_;  // 声明成员变量is_shooting_(射击状态标志,true表示正在射击)
    int consecutive_miss_count_;  // 声明成员变量consecutive_miss_count_(连续未命中计数,记录连续未击中的次数)
    int consecutive_miss_count_for_disable_;  // 连续未达标计数（用于取消自瞄，阈值10）
    int consecutive_miss_count_for_swing_;  // 连续未达标计数（用于开始摆动，阈值3）
    int initial_enemy_health_;  // 声明成员变量initial_enemy_health_(存储敌人初始血量)
    rclcpp::Time last_hit_time_;  // 声明成员变量last_hit_time_(记录最后一次击中的时间)
    rclcpp::Time shoot_start_time_;  // 声明成员变量shoot_start_time_(记录射击开始时间)
    rclcpp::Time consecutive_miss_start_time_;  // 声明成员变量consecutive_miss_start_time_(记录连续未命中的开始时间)
    bool is_swinging_;  // 声明成员变量is_swinging_(摆动瞄准标志,true表示正在摆动)
        double swing_base_yaw_;  // 声明成员变量swing_base_yaw_(摆动瞄准的基准偏航角)
        double swing_direction_;  // 摆动方向（1或-1）
        ConfigManager* config_manager_;  // 配置管理器指针
        double swing_range_;  // 摆动范围
        double swing_speed_;  // 摆动速度
        int consecutive_miss_threshold_disable_;  // 禁用自动瞄准的连续未命中阈值
        int consecutive_miss_threshold_swing_;  // 开始摆动瞄准的连续未命中阈值
        int hit_threshold_;  // 血量减少阈值
    
    public:
            ShootController(ConfigManager* config_manager) : auto_aim_enabled_(false), auto_aim_disabled_(false),  // 声明构造函数ShootController(初始化射击控制器,设置所有状态为初始值)
                                                  is_shooting_(false), consecutive_miss_count_(0),  // 初始化射击状态和连续未命中计数器
                                                  consecutive_miss_count_for_disable_(0), consecutive_miss_count_for_swing_(0),  // 初始化连续未命中计数器为0
                                                  initial_enemy_health_(-1), is_swinging_(false),  // 初始值设为-1(表示尚未发现敌人)
                                                  swing_base_yaw_(0), swing_direction_(1), config_manager_(config_manager) {
                    // 从ConfigManager获取配置
                    swing_range_ = 6.0;
                    consecutive_miss_threshold_disable_ = 10;
                    consecutive_miss_threshold_swing_ = 3;
                    hit_threshold_ = -10;
                    swing_speed_ = 10.0;
                    
                    if (config_manager && config_manager->isLoaded()) {
                            const auto& config = config_manager->getConfig();
                            swing_range_ = config.swing.range;
                            consecutive_miss_threshold_disable_ = config.threshold.consecutive_miss_disable;
                            consecutive_miss_threshold_swing_ = config.threshold.consecutive_miss_swing;
                            hit_threshold_ = config.threshold.hit;
                            swing_speed_ = config.swing.speed;
                    }
            }  // 初始化摆动基准角度为0,摆动方向为1

        void startShooting(int initial_health, rclcpp::Time current_time) {
                is_shooting_ = true;  // 进入射击状态并启动自动瞄准
                shoot_start_time_ = current_time;  // 将shoot_start_time_赋值为current_time
                auto_aim_enabled_ = true;  // 允许自动瞄准(射击控制器开始追踪目标)
                auto_aim_disabled_ = false;  // 将auto_aim_disabled_设置为false
                consecutive_miss_count_ = 0;  // 击中目标后重置(表示瞄准准确)
                consecutive_miss_count_for_disable_ = 0;
                consecutive_miss_count_for_swing_ = 0;
                last_hit_time_ = current_time;  // 将last_hit_time_赋值为current_time
                consecutive_miss_start_time_ = current_time;  // 记录开始连续未命中的时间(用于计算禁用条件)
                initial_enemy_health_ = initial_health;  // 设置敌方初始血量为initial_health(记录敌人刚开始的血量)
                RCLCPP_INFO(rclcpp::get_logger("ShootController"), "启用自瞄，初始血量: %d", initial_health);  // 输出日志信息(启用自动瞄准,显示初始血量)
    }

        void stopShooting() {  // 声明成员函数stopShooting(停止射击,设置is_shooting_为false,禁用自动瞄准)
                is_shooting_ = false;  // 退出射击状态并停止自动瞄准
                auto_aim_enabled_ = false;  // 禁止自动瞄准(连续未命中次数过多时)
                auto_aim_disabled_ = false;
                is_swinging_ = false;  // 将is_swinging_设置为false
                consecutive_miss_count_ = 0;
                consecutive_miss_count_for_disable_ = 0;
                consecutive_miss_count_for_swing_ = 0;
    }

        void enableAutoAim() { auto_aim_enabled_ = true; }  // 声明成员函数enableAutoAim(启用自动瞄准,设置auto_aim_enabled_为true)
        void disableAutoAim() {  // 声明成员函数disableAutoAim(禁用自动瞄准,设置auto_aim_disabled_为true)
                auto_aim_enabled_ = false;
                auto_aim_disabled_ = true;  // 将auto_aim_disabled_设置为true
        consecutive_miss_count_for_swing_ = 0;  // 重置摆动计数，让硬编码模式独立统计
    }

        bool isShooting() const { return is_shooting_; }  // 声明成员函数isShooting(判断是否正在射击,返回is_shooting_的值)
        bool isAutoAimEnabled() const { return auto_aim_enabled_; }  // 声明成员函数isAutoAimEnabled(判断自动瞄准是否启用,返回auto_aim_enabled_的值)
        bool isAutoAimDisabled() const { return auto_aim_disabled_; }  // 返回结果
        int getInitialEnemyHealth() const { return initial_enemy_health_; }  // 返回结果

    // 检查血量变化
        bool checkHealthChange(int health_change, int threshold, rclcpp::Time current_time) {
        // 血量变化小于阈值才算明显变化（表示血量减少）
                if (health_change < threshold) {  // 如果小于阈值,触发相应逻辑
                        last_hit_time_ = current_time;
                        consecutive_miss_count_ = 0;
                        consecutive_miss_count_for_disable_ = 0;
                        consecutive_miss_count_for_swing_ = 0;
                        consecutive_miss_start_time_ = current_time;
                        return true;  // 返回true表示操作成功或条件满足
                } else {  // 否则(未进入摆动瞄准模式)
                        consecutive_miss_count_++;  // 连续未命中计数器加1
                        consecutive_miss_count_for_disable_++;  // 禁用自动瞄准的连续未命中计数器加1
                        consecutive_miss_count_for_swing_++;  // 摆动瞄准的连续未命中计数器加1
                        if (consecutive_miss_count_ == 1) {  // 如果条件满足,执行对应操作
                                consecutive_miss_start_time_ = current_time;
            }
                        return false;  // 返回false表示操作失败或条件不满足
        }
    }

    // 检查是否应该禁用自瞄
        bool shouldDisableAutoAim(rclcpp::Time current_time) {
                if (!auto_aim_enabled_ || auto_aim_disabled_) return false;  // 返回false(条件不满足)

        // 记录到连续5组目标单位血量变化未达到明显标准
                bool too_many_misses = consecutive_miss_count_for_disable_ >= consecutive_miss_threshold_disable_;  // 检查连续未命中次数是否超过阈值(用于判断是否禁用自动瞄准)

                return too_many_misses;  // 返回是否连续未命中次数过多
    }

    // 检查是否应该开始摆动
        bool shouldStartSwinging(rclcpp::Time current_time) {
                if (!auto_aim_disabled_ || is_swinging_) return false;
    
        // 记录到超过连续3组目标单位血量变化未达到明显标准
                bool too_many_misses = consecutive_miss_count_for_swing_ >= consecutive_miss_threshold_swing_;  // 检查连续未命中次数是否超过阈值(用于判断是否开始摆动瞄准)
    
                return too_many_misses;
    }

    // 获取连续未达标计数（用于摆动）
        int getConsecutiveMissCountForSwing() const {  // 声明成员函数getConsecutiveMissCountForSwing(获取触发摆动瞄准的连续未命中次数阈值)
                return consecutive_miss_count_for_swing_;  // 返回结果
    }

    // 开始摆动
        void startSwinging(double base_yaw, rclcpp::Time current_time) {
                is_swinging_ = true;  // 将is_swinging_设置为true
                swing_base_yaw_ = base_yaw;  // 设置摆动基准角度为base_yaw(初始化摆动的中心角度)
                swing_direction_ = 1;  // 将swing_direction_赋值为1
    }

    // 停止摆动
        void stopSwinging(rclcpp::Time current_time) {
                is_swinging_ = false;
                consecutive_miss_count_ = 0;
                consecutive_miss_count_for_disable_ = 0;
                consecutive_miss_count_for_swing_ = 0;
                last_hit_time_ = current_time;
                consecutive_miss_start_time_ = current_time;
    }

    // 获取摆动角度
        double getSwingYaw(double dt) {
                if (!is_swinging_) return 0;

                swing_base_yaw_ += swing_direction_ * swing_speed_ * dt;  // 更新摆动基准角度(向摆动方向转动swing_speed_*dt度)
                if (swing_base_yaw_ > swing_range_) {  // 如果条件满足,执行对应操作
                        swing_base_yaw_ = swing_range_;  // 限制摆动范围(不超过swing_range_的正上限)
                        swing_direction_ = -1;  // 将swing_direction_赋值为-1
                } else if (swing_base_yaw_ < -swing_range_) {  // 如果条件满足,执行对应操作
                        swing_base_yaw_ = -swing_range_;  // 限制摆动范围(不超过-swing_range_的负下限)
                        swing_direction_ = 1;
        }

                return swing_base_yaw_;  // 返回结果
    }

        bool isSwinging() const { return is_swinging_; }  // 返回结果
};

// ==================== TensorRT日志记录器 ====================
class Logger : public nvinfer1::ILogger {  // 日志类(TensorRT日志输出)
        void log(Severity severity, const char* msg) noexcept override {  // 日志输出函数(用于记录TensorRT的调试信息)
                if (severity <= Severity::kWARNING) {  // 将<赋值为Severity::kWARNING) {
        }
    }
};

// ==================== CenterNet装甲板检测器 ====================
class CenterNetDetector {  // 目标检测器类(TensorRT装甲板检测)
private:
        int player_id_;
        cv::Mat latest_frame_;  // 存储最新接收的相机图像(用于目标检测)
        std::unique_ptr<nvinfer1::IRuntime> runtime_;
        std::unique_ptr<nvinfer1::ICudaEngine> engine_;  // TensorRT推理引擎(包含优化后的神经网络模型)
        std::unique_ptr<nvinfer1::IExecutionContext> context_;  // TensorRT执行上下文(用于执行推理)
    
        void* gpu_buffers_[4];  // GPU内存指针数组(存储4个TensorRT缓冲区的地址:输入+输出)
        float* cpu_fmap_;  // CPU上的特征图内存(用于从GPU拷贝推理结果,大小=num_classes_×output_h_×output_w_)
        float* cpu_reg_;  // CPU上的回归框数据(用于存储检测结果的位置)
        float* cpu_wh_;  // CPU上的宽高数据(用于存储检测框的宽度和高度)
    
        int input_h_;  // 声明成员变量input_h_(存储输入图像高度)
        int input_w_;  // 声明成员变量input_w_(存储输入图像宽度)
        int output_h_;  // 声明成员变量output_h_(存储输出特征图高度)
        int output_w_;  // 声明成员变量output_w_(存储输出特征图宽度)
        int num_classes_;  // 声明成员变量num_classes_(存储目标类别数量)
    
        int target_class_start_;  // 声明成员变量target_class_start_(存储目标类别起始索引)
        int target_class_end_;  // 声明成员变量target_class_end_(存储目标类别结束索引)
    
        float conf_threshold_;  // 置信度阈值(低于此值的目标将被过滤掉)
        const int K = 8;  // K值8表示需要4个3D-2D点对来求解相机姿态
        Logger logger_;  // TensorRT日志记录器
        ConfigManager* config_manager_;  // 配置管理器
    
    // 相机和弹道参数
        double camera_fx_;
        double camera_fy_;
        double camera_cx_;
        double camera_cy_;
        double bullet_speed_;
        double gravity_;
    
    // 目标位置历史（用于运动预测）
    // 优化评估：适合使用RingBuffer<cv::Point2f, 5>替代deque
    // 理由：固定大小（MAX_HISTORY_SIZE），用于运动预测，不需要频繁插入中间元素
    std::deque<cv::Point2f> target_position_history_;  // 声明双端队列target_position_history_(存储目标位置历史记录)
        static const size_t MAX_HISTORY_SIZE = 5;  // 最多保存5个历史记录(用于目标位置预测)
    
    // 滤波器
        MovingAverageFilter distance_filter_;  // 声明距离滤波器distance_filter_(用于平滑PnP距离)
        ExponentialSmoothingFilter yaw_filter_;  // 声明偏航角滤波器yaw_filter_(用于平滑yaw角度)
        ExponentialSmoothingFilter pitch_filter_;  // 声明俯仰角滤波器pitch_filter_(用于平滑pitch角度)

public:
        struct DetectedArmor {  // 检测到的装甲板信息
                cv::Point2f center;  // 声明2D点变量center(存储装甲板中心点)
                cv::RotatedRect leftBar;  // 声明旋转矩形leftBar(存储左装甲板条)
                cv::RotatedRect rightBar;  // 声明旋转矩形rightBar(存储右装甲板条)
                cv::Rect armorBbox;  // 声明矩形armorBbox(存储装甲板边界框)
                float score;  // 声明浮点数变量score(存储检测置信度,范围0-1)
                int class_id;  // 声明整数变量class_id(存储类别ID,0-5表示不同类别)
        };
    
        CenterNetDetector(int player_id, ConfigManager* config_manager) : player_id_(player_id), config_manager_(config_manager) {  // 声明构造函数CenterNetDetector(初始化检测器和滤波器,降低滤波器响应速度以适应1ms调用周期)
                if (player_id == 1) {  // 如果条件满足,执行对应操作
                        target_class_start_ = 0;  // Player1红队识别red (class 0)
                        target_class_end_ = 0;
                } else {  // 否则(检测失败时)
                        target_class_start_ = 1;  // Player2蓝队识别blue (class 1)
                        target_class_end_ = 1;
        }
        
                conf_threshold_ = 0.3f;  // 将conf_threshold_赋值为0.3f(设置置信度阈值)
                
                // 从ConfigManager获取配置参数（如果有）
                int distance_window = 5;
                double yaw_alpha = 0.1;
                double pitch_alpha = 0.1;
                
                // 相机和弹道参数（硬编码）
                camera_fx_ = 554.26;
                camera_fy_ = 554.26;
                camera_cx_ = 320.0;
                camera_cy_ = 240.0;
                bullet_speed_ = 23.0;
                gravity_ = 5.0;
                
                if (config_manager && config_manager->isLoaded()) {
                        const auto& config = config_manager->getConfig();
                        distance_window = config.filter.distance_window;
                        yaw_alpha = config.filter.yaw_alpha;
                        pitch_alpha = config.filter.pitch_alpha;
                        // 相机和弹道参数暂不支持配置，使用硬编码值
                }
                
                // 使用配置参数初始化滤波器
                distance_filter_ = MovingAverageFilter(distance_window);
                yaw_filter_ = ExponentialSmoothingFilter(yaw_alpha);
                pitch_filter_ = ExponentialSmoothingFilter(pitch_alpha);
                
                initTensorRT();  // 初始化TensorRT推理引擎
    }
    
        ~CenterNetDetector() {
                cleanup();  // 调用cleanup函数(清理TensorRT资源)
    }
    
        void updateFrame(const cv::Mat& frame) {  // OpenCV图像操作
                latest_frame_ = frame.clone();  // 将latest_frame_赋值为frame.clone()

                // 每次收到图像都进行检测并显示窗口(让检测窗口一直开着)
                std::vector<DetectedArmor> armors = detectArmors();  // 调用检测函数(检测装甲板)

                // 可视化检测结果
                cv::Mat display = latest_frame_.clone();  // 创建图像副本用于显示
                for (const auto& armor : armors) {
                        cv::rectangle(display, armor.armorBbox, cv::Scalar(0, 255, 0), 2);  // 绘制矩形(在图像上绘制装甲板边界框)
                        cv::circle(display, armor.center, 5, cv::Scalar(0, 0, 255), -1);  // 绘制圆形(在图像上绘制装甲板中心点)
                        std::string label = "C" + std::to_string(armor.class_id) + ":" + std::to_string(armor.score).substr(0, 4);  // 创建标签字符串(类别:置信度,例如"C3:0.95")
                        cv::putText(display, label, cv::Point(armor.armorBbox.x, armor.armorBbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);  // 绘制文字(在装甲板上方显示类别和置信度)
                }
                std::string label = "Armors: " + std::to_string(armors.size());
                cv::putText(display, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);  // 绘制文字(在图像上显示检测信息)
                cv::imshow("CenterNet Detection", display);  // 显示图像(显示检测结果)
                cv::waitKey(1);  // 等待按键(1毫秒,用于刷新显示)
    }
    
        std::vector<DetectedArmor> detectArmors() {
                std::vector<DetectedArmor> detected_armors;  // 检测到的装甲板列表(按置信度排序)
                if (latest_frame_.empty()) return detected_armors;
        
        // 保存原始图像尺寸
                int orig_h = latest_frame_.rows;  // 将orig_h赋值为latest_frame_.rows
                int orig_w = latest_frame_.cols;  // 将orig_w赋值为latest_frame_.cols
        
        // 预处理（性能优化点：考虑使用cv::dnn::blobFromImage简化）
                cv::Mat processed = preprocess(latest_frame_);
        
        // GPU推理
                auto detections = infer(processed);  // 将detections赋值为infer(processed)
        
        // 后处理
                for (const auto& det : detections) {  // 遍历所有检测结果(处理每个检测到的装甲板)
                        if (det.class_id >= target_class_start_ && det.class_id <= target_class_end_) {  // 将>赋值为target_class_start_ && det.class_id <
                // 坐标映射回原始图像尺寸
                                float mapped_x = det.x / input_w_ * orig_w;  // 输入图像宽度(从TensorRT引擎获取)
                                float mapped_y = det.y / input_h_ * orig_h;  // 输入图像高度(从TensorRT引擎获取)
                                float mapped_w = det.w / input_w_ * orig_w;
                                float mapped_h = det.h / input_h_ * orig_h;
                                float mapped_cx = det.center_x / input_w_ * orig_w;
                                float mapped_cy = det.center_y / input_h_ * orig_h;
                
                                DetectedArmor armor;  // 单个装甲板的信息(类别、位置、角度、置信度)
                                armor.center = cv::Point2f(mapped_cx, mapped_cy);  // OpenCV变量
                                armor.armorBbox = cv::Rect(mapped_x, mapped_y, mapped_w, mapped_h);
                                armor.leftBar = cv::RotatedRect(cv::Point2f(mapped_x, mapped_cy),
                                                                                                cv::Size2f(mapped_h/3, mapped_h), 0);  // 创建旋转矩形(设置装甲板条的尺寸和角度)
                                armor.rightBar = cv::RotatedRect(cv::Point2f(mapped_x + mapped_w, mapped_cy),
                                                                                                  cv::Size2f(mapped_h/3, mapped_h), 0);  // 创建旋转矩形(设置装甲板条的尺寸和角度)
                                armor.score = det.score;  // 保存检测置信度(用于显示和判断)
                                armor.class_id = det.class_id;  // 保存类别ID(用于显示)
                                detected_armors.push_back(armor);
            }
        }
        
                // D2C2两分类模型：按置信度排序，只保留最高置信度的目标
                        std::sort(detected_armors.begin(), detected_armors.end(),
                                  [](const DetectedArmor& a, const DetectedArmor& b) {
                                          return a.score > b.score;
                                  });

                // 每帧只保留最高置信度的目标，其余不识别
                if (!detected_armors.empty()) {
                        detected_armors.resize(1);
                }

        // 不在这里显示窗口,避免重复显示(窗口显示移到updateFrame中)
                return detected_armors;
    }
    
        std::tuple<double, double> imageToGimbalAngles(cv::Point2f image_point) {  // 声明函数imageToGimbalAngles(将图像坐标转换为云台角度)
                double dx = image_point.x - camera_cx_;  // 计算目标相对于图像中心的X偏移
                double dy = image_point.y - camera_cy_;  // 计算目标相对于图像中心的Y偏移
                double yaw_offset = std::atan2(dx, camera_fx_) * 180.0 / CV_PI;  // 将X偏移转换为偏航角偏移(使用atan2,单位度)
                double pitch_offset = -std::atan2(dy, camera_fx_) * 180.0 / CV_PI;  // 将Y偏移转换为俯仰角偏移(负号因为图像坐标系Y向下,使用统一焦距)
                return {yaw_offset, pitch_offset};  // 返回偏航角和俯仰角偏移
    }
    
        std::tuple<double, double> calculateBallisticCompensation(  // 声明函数calculateBallisticCompensation(计算弹道补偿)
                double yaw_offset, double pitch_offset, double distance,  // 声明函数参数(偏航角偏移、俯仰角偏移、距离)
                double current_yaw, double current_pitch) {
                double t_flight = distance / bullet_speed_;  // 计算子弹飞行时间(距离除以速度)
                double gravity_drop = 0.5 * gravity_ * t_flight * t_flight;  // 计算重力下落距离(0.5*g*t²)
                
                // 限制 asin 参数范围(避免小距离时不稳定)
                double drop_ratio = gravity_drop / distance;  // 计算下落比例
                if (drop_ratio > 0.99) drop_ratio = 0.99;  // 限制最大比例为0.99
                if (drop_ratio < -0.99) drop_ratio = -0.99;  // 限制最小比例为-0.99
                
                double pitch_compensation = std::asin(drop_ratio) * 180.0 / CV_PI;  // 计算俯仰角补偿(使用asin,单位度)
                
                // 只滤波偏移量,不使用 current_yaw 作为基础(避免点位预设角度干扰)
                double filtered_yaw_offset = yaw_filter_.update(yaw_offset);
                double filtered_pitch_offset = pitch_filter_.update(pitch_offset + pitch_compensation);
                
                // 返回绝对角度(基础角度 + 滤波后的偏移)
                return {current_yaw + filtered_yaw_offset, current_pitch + filtered_pitch_offset};  // 返回绝对偏航角和俯仰角
    }
    
        std::tuple<double, cv::Point3f> solvePnPDistance(const cv::Rect& armor_bbox) {
                // 输入有效性检查
                if (armor_bbox.width <= 0 || armor_bbox.height <= 0) {
                        return {3.0, cv::Point3f(0, 0, 3.0)};
                }

                std::vector<cv::Point2f> corners_2d = {
                        cv::Point2f(armor_bbox.x, armor_bbox.y),
                        cv::Point2f(armor_bbox.x + armor_bbox.width, armor_bbox.y),
                        cv::Point2f(armor_bbox.x, armor_bbox.y + armor_bbox.height),
                        cv::Point2f(armor_bbox.x + armor_bbox.width, armor_bbox.y + armor_bbox.height)
                };
                std::vector<cv::Point3f> corners_3d = {
                        cv::Point3f(-ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0),
                        cv::Point3f( ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0),
                        cv::Point3f(-ARMOR_WIDTH/2,  ARMOR_HEIGHT/2, 0),
                        cv::Point3f( ARMOR_WIDTH/2,  ARMOR_HEIGHT/2, 0)
                };

                cv::Mat camera_matrix = (cv::Mat_<double>(3,3) <<
                        camera_fx_, 0, camera_cx_,
                        0, camera_fy_, camera_cy_,
                        0, 0, 1);
                cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
                cv::Mat rvec, tvec;
                
                // 性能优化点：考虑缓存solvePnP结果，避免重复计算
                bool success = cv::solvePnP(corners_3d, corners_2d, camera_matrix, dist_coeffs, rvec, tvec);
                if (success && !tvec.empty()) {
                        double distance = tvec.at<double>(2);
                        distance_filter_.update(distance);
                        cv::Point3f position(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
                        return {distance_filter_.getFiltered(), position};
        }
                return {3.0, cv::Point3f(0, 0, 3.0)};
    }
    
        cv::Point2f predictTargetPosition(double dt = 0.01) {  // 声明预测目标位置函数(预测目标在dt秒后的位置)
                if (target_position_history_.size() < 2) {
                        // 历史记录不足,直接返回最新位置(避免返回0,0导致大幅偏移)
                        if (target_position_history_.empty()) {
                                return cv::Point2f(camera_cx_, camera_cy_);  // 没有历史记录时返回图像中心
                        }
                        return target_position_history_.back();  // 只有1个记录时返回它
                }                
                // 使用更多历史帧进行预测(减少噪声影响)
                int n = std::min(5, (int)target_position_history_.size());  // 确定使用的帧数(最多5帧)
                double vx = 0, vy = 0;  // 声明速度变量vx,vy(用于累加速度)
                
                for (int i = 1; i < n; i++) {  // 遍历历史帧,计算平均速度
                        double dx = target_position_history_[i].x - target_position_history_[i-1].x;  // 计算X方向位移
                        double dy = target_position_history_[i].y - target_position_history_[i-1].y;  // 计算Y方向位移
                        vx += dx / dt;  // 累加X方向速度
                        vy += dy / dt;  // 累加Y方向速度
                }
                
                vx /= (n - 1);  // 计算平均X速度
                vy /= (n - 1);  // 计算平均Y速度
                
                auto& latest = target_position_history_.back();  // 获取最新的目标位置
                double predicted_x = latest.x + vx * dt;  // 预测X坐标(当前位置+速度*时间)
                double predicted_y = latest.y + vy * dt;  // 预测Y坐标(当前位置+速度*时间)
                return cv::Point2f(predicted_x, predicted_y);  // 返回预测位置
    }
    
        void updateTargetHistory(const cv::Point2f& center) {  // 声明函数updateTargetHistory(更新目标历史记录,用于平滑追踪)
                target_position_history_.push_back(center);  // 将新位置添加到历史记录
                while (target_position_history_.size() > MAX_HISTORY_SIZE) {  // 循环删除直到队列长度不超过最大值
                        target_position_history_.pop_front();
                }
    }

private:
        struct Detection {  // 检测结果(位置、分数、类别)
                float x, y, w, h;  // 声明浮点变量x,y,w,h(存储检测框的坐标和尺寸)
                float score;  // 声明浮点变量score(存储检测置信度)
                int class_id;  // 声明整型变量class_id(存储类别ID)
                float center_x, center_y;  // 声明浮点变量center_x,center_y(存储中心点坐标)
        };
    
        void initTensorRT() {
                std::cout << "Initializing TensorRT..." << std::endl;  // 输出初始化信息(打印TensorRT初始化开始)
                std::string engine_path = "/home/robot/competition/models/centernet_r18d2c2_fp16.engine";  // TensorRT模型文件路径(r18d2c2两分类版本)
                std::ifstream file(engine_path, std::ios::binary);
                if (!file.is_open()) {  // 如果条件满足,执行对应操作
                        throw std::runtime_error("Failed to open engine file: " + engine_path);
        }
                file.seekg(0, std::ios::end);  // 移动文件指针到末尾,获取文件大小
                size_t size = file.tellg();  // 将size赋值为file.tellg()
                if (size == 0) {
                        throw std::runtime_error("Engine file is empty: " + engine_path);
                }
                file.seekg(0, std::ios::beg);  // 移动文件指针到开头(准备读取数据)
                std::vector<char> engine_data(size);
                file.read(engine_data.data(), size);
                file.close();  // 关闭文件(释放文件资源)
        
                runtime_.reset(nvinfer1::createInferRuntime(logger_));
                if (!runtime_) {
                        throw std::runtime_error("Failed to create TensorRT runtime");
                }
                
                engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
                if (!engine_) {
                        throw std::runtime_error("Failed to deserialize CUDA engine");
                }
                
                context_.reset(engine_->createExecutionContext());
                if (!context_) {
                        throw std::runtime_error("Failed to create execution context");
                }
        
                auto input_dims = engine_->getTensorShape("input");
                input_h_ = input_dims.d[2];
                input_w_ = input_dims.d[3];
        
                auto fmap_dims = engine_->getTensorShape("fmap");
                num_classes_ = fmap_dims.d[1];  // 目标类别数量(从TensorRT引擎获取)
                output_h_ = fmap_dims.d[2];  // 将output_h_赋值为fmap_dims.d[2]
                output_w_ = fmap_dims.d[3];  // 将output_w_赋值为fmap_dims.d[3]
        
                std::cout << "Model: " << input_w_ << "x" << input_h_ << " -> "  // 输出模型信息(打印输入输出尺寸和类别数)
                                    << output_w_ << "x" << output_h_ << " (" << num_classes_ << " classes)" << std::endl;
        
                // CUDA内存分配错误检查
                cudaError_t err;
                err = cudaMalloc(&gpu_buffers_[0], 3 * input_h_ * input_w_ * sizeof(float));
                if (err != cudaSuccess) {
                        throw std::runtime_error("CUDA malloc failed for input buffer: " + std::string(cudaGetErrorString(err)));
                }
                err = cudaMalloc(&gpu_buffers_[1], num_classes_ * output_h_ * output_w_ * sizeof(float));
                if (err != cudaSuccess) {
                        cudaFree(gpu_buffers_[0]);  // 清理已分配的内存
                        throw std::runtime_error("CUDA malloc failed for fmap buffer: " + std::string(cudaGetErrorString(err)));
                }
                err = cudaMalloc(&gpu_buffers_[2], 2 * output_h_ * output_w_ * sizeof(float));
                if (err != cudaSuccess) {
                        cudaFree(gpu_buffers_[0]);
                        cudaFree(gpu_buffers_[1]);  // 清理已分配的内存
                        throw std::runtime_error("CUDA malloc failed for reg buffer: " + std::string(cudaGetErrorString(err)));
                }
                err = cudaMalloc(&gpu_buffers_[3], 2 * output_h_ * output_w_ * sizeof(float));
                if (err != cudaSuccess) {
                        cudaFree(gpu_buffers_[0]);
                        cudaFree(gpu_buffers_[1]);
                        cudaFree(gpu_buffers_[2]);  // 清理已分配的内存
                        throw std::runtime_error("CUDA malloc failed for wh buffer: " + std::string(cudaGetErrorString(err)));
                }
        
                // CPU内存分配（使用智能指针自动管理）
                cpu_fmap_ = new (std::nothrow) float[num_classes_ * output_h_ * output_w_];
                if (!cpu_fmap_) {
                        throw std::runtime_error("Failed to allocate CPU memory for fmap");
                }
                cpu_reg_ = new (std::nothrow) float[2 * output_h_ * output_w_];  // 在CPU上分配cpu_reg_的内存(大小=2 * output_h_ * output_w_个float元素)
                if (!cpu_reg_) {
                        delete[] cpu_fmap_;
                        throw std::runtime_error("Failed to allocate CPU memory for reg");
                }
                cpu_wh_ = new (std::nothrow) float[2 * output_h_ * output_w_];  // 在CPU上分配cpu_wh_的内存(大小=2 * output_h_ * output_w_个float元素)
                if (!cpu_wh_) {
                        delete[] cpu_fmap_;
                        delete[] cpu_reg_;
                        throw std::runtime_error("Failed to allocate CPU memory for wh");
                }
        
                std::cout << "TensorRT initialized!" << std::endl;  // 输出完成信息(打印TensorRT初始化完成)
    }
    
        cv::Mat preprocess(const cv::Mat& frame) {  // OpenCV图像矩阵(preprocess(const cv::Mat& frame) {)
                cv::Mat resized;  // 存储调整大小后的图像(缩放到input_h_×input_w_,用于TensorRT输入)
                cv::resize(frame, resized, cv::Size(input_w_, input_h_));  // 调整图像大小(缩放到TensorRT要求的输入尺寸)
                cv::Mat rgb;  // 存储BGR转RGB后的图像(用于TensorRT推理)
                cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);  // 转换颜色空间(BGR转RGB)
                rgb.convertTo(rgb, CV_32F);  // 转换数据类型(8位转32位浮点)
                std::vector<cv::Mat> channels(3);
                cv::split(rgb, channels);  // 分离通道(将RGB图像分离为3个通道)
                channels[0] = (channels[0] - 123.675) / 58.395;  // 将channels[0]赋值为(channels[0] - 123.675) / 58.395
                channels[1] = (channels[1] - 116.28) / 57.12;  // 将channels[1]赋值为(channels[1] - 116.28) / 57.12
                channels[2] = (channels[2] - 103.53) / 57.375;  // 将channels[2]赋值为(channels[2] - 103.53) / 57.375
                cv::merge(channels, rgb);  // 合并通道(将3个通道合并为RGB图像)
                return rgb;  // 返回结果
    }
    
        std::vector<Detection> infer(const cv::Mat& processed) {
                std::vector<cv::Mat> channels(3);
                cv::split(processed, channels);  // 分离通道(将RGB图像分离为3个通道)
                // 性能优化点：考虑预分配内存，避免每次调用都分配
                std::vector<float> input_data(3 * input_h_ * input_w_);  // 将BGR图像转为3通道浮点数据(用于TensorRT推理)
                for (int c = 0; c < 3; c++) {  // 循环3次,处理RGB三个通道
                        memcpy(input_data.data() + c * input_h_ * input_w_,  // 复制图像数据(将通道数据拷贝到输入数组)
                                      channels[c].ptr<float>(), input_h_ * input_w_ * sizeof(float));  // 获取通道数据指针(用于拷贝数据)
        }
                // 性能优化点：考虑使用cudaMemcpyAsync异步传输
                cudaMemcpy(gpu_buffers_[0], input_data.data(),
                                      3 * input_h_ * input_w_ * sizeof(float), cudaMemcpyHostToDevice);  // 拷贝数据(CUDA数据传输)
                context_->executeV2(gpu_buffers_);  // GPU推理（主要性能瓶颈）
                cudaMemcpy(cpu_fmap_, gpu_buffers_[1],
                                      num_classes_ * output_h_ * output_w_ * sizeof(float), cudaMemcpyDeviceToHost);  // 拷贝数据(CUDA数据传输)
                cudaMemcpy(cpu_reg_, gpu_buffers_[2],
                                      2 * output_h_ * output_w_ * sizeof(float), cudaMemcpyDeviceToHost);  // 拷贝数据(CUDA数据传输)
                cudaMemcpy(cpu_wh_, gpu_buffers_[3],
                                      2 * output_h_ * output_w_ * sizeof(float), cudaMemcpyDeviceToHost);  // 拷贝数据(CUDA数据传输)
                return decodeDetections();  // 返回结果
    }
    
        std::vector<Detection> decodeDetections() {
                std::vector<Detection> detections;  // 存储TensorRT网络的检测输出(用于后续处理)
                struct ScoreIndex { float score; int cls; int idx; };  // 分数索引(用于排序)
                std::vector<ScoreIndex> all_scores;  // 声明动态数组all_scores(存储所有检测分数)
        
                for (int cls = 0; cls < num_classes_; cls++) {  // 循环遍历所有类别(num_classes_个类别,比如装甲板编号)
                        for (int i = 0; i < output_h_ * output_w_; i++) {  // 循环遍历所有像素点(output_h_ × output_w_个点)
                                float score = cpu_fmap_[cls * output_h_ * output_w_ + i];
                                if (score > 0.01f) {  // 如果条件满足,执行对应操作
                                        all_scores.push_back({score, cls, i});  // 添加元素到数组末尾
                }
            }
        }
        
                std::sort(all_scores.begin(), all_scores.end(),  // 获取迭代器起始位置
                          [](const ScoreIndex& a, const ScoreIndex& b) { return a.score > b.score; });  // 返回结果
        
                int top_k = std::min(K, (int)all_scores.size());  // 将top_k赋值为std::min(K, (int)all_scores.size())
                for (int k = 0; k < top_k; k++) {  // 循环top_k次,处理top_k个最高分的点
                        float score = all_scores[k].score;  // 将score赋值为all_scores[k].score
                        if (score < conf_threshold_) continue;  // 如果置信度低于阈值,跳过该检测点(只保留高置信度的检测结果)
            
                        int cls = all_scores[k].cls;  // 将cls赋值为all_scores[k].cls
                        int idx = all_scores[k].idx;  // 将idx赋值为all_scores[k].idx
                        int y = idx / output_w_;  // 将y赋值为idx / output_w_
                        int x = idx % output_w_;  // 将x赋值为idx % output_w_
            
                        float reg_x = cpu_reg_[idx];  // 将reg_x赋值为cpu_reg_[idx]
                        float reg_y = cpu_reg_[output_h_ * output_w_ + idx];  // 将reg_y赋值为cpu_reg_[output_h_ * output_w_ + idx]
                        float w = cpu_wh_[idx];  // 将w赋值为cpu_wh_[idx]
                        float h = cpu_wh_[output_h_ * output_w_ + idx];  // 将h赋值为cpu_wh_[output_h_ * output_w_ + idx]
            
                        float center_x = (x + reg_x) * 4;  // 将center_x赋值为(x + reg_x) * 4
                        float center_y = (y + reg_y) * 4;  // 将center_y赋值为(y + reg_y) * 4
            
                        Detection det;  // 声明检测对象det(存储单个检测结果)
                        det.x = center_x - w * 2;  // 将det.x赋值为center_x - w * 2
                        det.y = center_y - h * 2;  // 将det.y赋值为center_y - h * 2
                        det.w = w * 4;  // 将det.w赋值为w * 4
                        det.h = h * 4;  // 将det.h赋值为h * 4
                        det.score = score;  // 将det.score赋值为score
                        det.class_id = cls;  // 将det.class_id赋值为cls
                        det.center_x = center_x;  // 将det.center_x赋值为center_x
                        det.center_y = center_y;  // 将det.center_y赋值为center_y
                        detections.push_back(det);  // 添加元素到数组末尾
        }
                return detections;  // 返回结果
    }
    
        void cleanup() {
                if (gpu_buffers_[0]) cudaFree(gpu_buffers_[0]);
                if (gpu_buffers_[1]) cudaFree(gpu_buffers_[1]);
                if (gpu_buffers_[2]) cudaFree(gpu_buffers_[2]);
                if (gpu_buffers_[3]) cudaFree(gpu_buffers_[3]);
                if (cpu_fmap_) delete[] cpu_fmap_;
                if (cpu_reg_) delete[] cpu_reg_;  // 如果条件满足,执行对应操作
                if (cpu_wh_) delete[] cpu_wh_;  // 如果条件满足,执行对应操作
    }
};

// ==================== 主节点类 ====================
class GoNode : public rclcpp::Node {  // 定义GoNode类(ROS2主节点类,继承自rclcpp::Node,负责机器人的主控逻辑)
private:
    // ROS2订阅者
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr position_sub_;  // 声明位置订阅者(接收机器人位置数据)
    rclcpp::Subscription<tdt_interface::msg::ReceiveData>::SharedPtr real_angles_sub_;  // 声明角度订阅者(接收云台角度数据)
    rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr health_sub_;  // 声明血量订阅者(接收血量数据)
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;  // 声明相机订阅者(接收相机图像)
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr game_time_sub_;  // 声明游戏时间订阅者(接收游戏时间)

    // ROS2发布者
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr speed_pub_;  // 声明速度发布者(发送机器人速度指令)
    rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr angles_pub_;  // 声明角度发布者(发送目标角度和射击指令)

    // ROS2定时器
    rclcpp::TimerBase::SharedPtr control_timer_;  // 声明控制定时器(5ms周期,执行主控逻辑)
    rclcpp::TimerBase::SharedPtr shoot_timer_;  // 声明射击定时器(1ms周期,执行射击控制)
    rclcpp::TimerBase::SharedPtr position_adjust_timer_;  // 声明位置调整定时器(50ms周期,执行位置调整)
    rclcpp::TimerBase::SharedPtr health_check_timer_;  // 声明健康检查定时器(1秒周期,检查血量变化)
    rclcpp::TimerBase::SharedPtr swing_timer_;  // 声明摆动定时器(1ms周期,执行摆动瞄准)

    // 模块化组件
    std::unique_ptr<DataManager> data_manager_;  // 声明数据管理器智能指针(管理血量、位置、速度等数据)
    std::unique_ptr<PathNavigator> path_navigator_;  // 声明路径导航器智能指针(管理移动路径和卡住检测)
    std::unique_ptr<ShootController> shoot_controller_;  // 声明射击控制器智能指针(管理射击逻辑和自动瞄准)
    std::unique_ptr<CenterNetDetector> centernet_detector_;  // 声明目标检测器智能指针(使用TensorRT进行装甲板检测)
    std::unique_ptr<PerformanceMonitor> performance_monitor_;  // 声明性能监控器智能指针(监控推理延迟、FPS等性能指标)
    std::unique_ptr<ConfigManager> config_manager_;  // 声明配置管理器智能指针(管理游戏配置)

    // 路径点配置
    const std::vector<Waypoint>* waypoints_;  // 路径点数组(存储比赛中的导航路径)

    // 状态变量
    enum State { MOVING, SHOOTING };  // 定义状态枚举(移动状态和射击状态)
    bool is_precise_adjusting_ = false;  // 是否正在进行精确调整
    State state_;  // 声明状态变量state_(存储当前机器人状态)
    rclcpp::Time shooting_start_time_;  // 进入SHOOTING状态的时间
    int player_id_;
    double target_yaw_;  // 声明目标偏航角变量(存储目标左右转动角度)
    double target_pitch_;  // 声明目标俯仰角变量(存储目标上下转动角度)
    double current_yaw_;  // 声明成员变量current_yaw_(当前的偏航角)
    double current_pitch_;  // 声明成员变量current_pitch_(当前的俯仰角)

    // 自瞄相关
    cv::Point2f locked_target_center_;  // 声明锁定目标中心点变量(存储锁定的装甲板中心坐标)
    bool target_locked_;  // 声明成员变量target_locked_(目标锁定标志,true表示目标已锁定)
    int detection_fail_count_;  // 检测失败计数器(连续检测失败次数)
    static const int MAX_DETECTION_FAIL_COUNT = 10;  // 最大检测失败次数(超过后启用降级策略)

    // 被攻击处理相关
    bool is_under_attack_;  // 声明成员变量is_under_attack_(被攻击标志,true表示正在被攻击)
    std::stack<size_t> attack_waypoint_stack_;  // 被攻击点位栈（用于记录所有被攻击的点位，便于返回）
    
    // 看门狗机制（防止循环卡死）
    mutable std::mutex watchdog_mutex_;
    rclcpp::Time control_loop_last_update_;
    rclcpp::Time shoot_loop_last_update_;
    static constexpr int WATCHDOG_TIMEOUT_MS = 100;

    // 配置参数
    double max_speed_far_;
    double max_speed_near_;
    double max_speed_precise_;
    double speed_switch_distance_;
    double speed_switch_distance_fast_;
    double precise_mode_distance_;
    double arrival_threshold_fast_;
    double arrival_threshold_normal_;
    double precise_arrival_threshold_;
    double precise_adjust_speed_;
    double position_hold_threshold_;
    double swing_range_;
    double swing_speed_;
    int consecutive_miss_threshold_disable_;
    int consecutive_miss_threshold_swing_;
    int hit_threshold_;
    double health_history_duration_;
    double other_data_duration_;
    double move_timeout_;
    int distance_filter_window_;
    double yaw_filter_alpha_;
    double pitch_filter_alpha_;
    double camera_fx_;
    double camera_fy_;
    double camera_cx_;
    double camera_cy_;
    double bullet_speed_;
    double gravity_;

    // 静态辅助函数
        static int healthTopicIndexToInternalIndex(int topic_index, int player_id) {
                if (player_id == 1) {
                // Player1（蓝方）：敌方是索引6-11
                        if (topic_index >= 6 && topic_index <= 11) {
                return topic_index - 6;  // 6→0, 7→1, ..., 11→5
                }
                } else {
                // Player2（红方）：敌方是索引0-5
                        if (topic_index >= 0 && topic_index <= 5) {
                return topic_index;  // 0→0, 1→1, ..., 5→5
                }
                }
                return -1;  // 返回结果
        }

public:
        GoNode(int player_id) : Node("go_node"), player_id_(player_id), state_(MOVING),
                                                            target_yaw_(0), target_pitch_(0),  // 初始化射击控制器的目标角度和状态标志
                                                            current_yaw_(0), current_pitch_(0),  // 初始化当前角度为0度
                                                            target_locked_(false),  // 初始化目标锁定状态为false
                                                            detection_fail_count_(0),  // 初始化检测失败计数器为0
                                                            is_under_attack_(false),  // 初始化被攻击状态为false
                                                            is_precise_adjusting_(false),  // 初始化精确调整状态为false
                                                            waypoints_(nullptr) {
        // 根据玩家ID选择路径点配置
                if (player_id_ == 1) {
                        waypoints_ = &WAYPOINTS_PLAYER1;
                } else {  // 否则(无法识别类别)
                        waypoints_ = &WAYPOINTS_PLAYER2;
        }
        
                // 初始化配置管理器并加载配置
                config_manager_ = std::make_unique<ConfigManager>();
                std::string config_path = "/home/robot/competition/src/go/config/game_config.yaml";
                if (config_manager_->loadFromFile(config_path)) {
                        RCLCPP_INFO(this->get_logger(), "配置文件加载成功");
                } else {
                        RCLCPP_WARN(this->get_logger(), "配置文件加载失败，使用默认值");
                }

                // 从ConfigManager读取配置参数
                loadConfigParameters();
                
                // 验证配置参数
                if (!validateConfig()) {
                        RCLCPP_WARN(this->get_logger(), "配置验证失败，使用默认配置继续运行");
                }

                data_manager_ = std::make_unique<DataManager>(player_id, health_history_duration_, other_data_duration_);  // 将data_manager_赋值为std::make_unique<DataManager>(player_id, health_history_duration_, other_data_duration_)
                path_navigator_ = std::make_unique<PathNavigator>(player_id, config_manager_.get());  // 将path_navigator_赋值为std::make_unique<PathNavigator>(player_id, config_manager_)
                shoot_controller_ = std::make_unique<ShootController>(config_manager_.get());  // 将shoot_controller_赋值为std::make_unique<ShootController>(config_manager_)
                centernet_detector_ = std::make_unique<CenterNetDetector>(player_id, config_manager_.get());  // 将centernet_detector_赋值为std::make_unique<CenterNetDetector>(player_id, config_manager_)
                performance_monitor_ = std::make_unique<PerformanceMonitor>(100, this->get_logger());  // 初始化性能监控器(最多保存100个样本)

                std::string position_topic = (player_id == 1) ? "/position_player_1" : "/position_player_2";  // 根据玩家ID选择位置话题(玩家1或玩家2)
                std::string angles_topic = (player_id == 1) ? "/real_angles_player_1" : "/real_angles_player_2";  // 根据玩家ID选择角度话题

                position_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(  // 创建位置订阅者(订阅position_topic话题,回调函数为positionCallback,队列大小为10)
                        position_topic, 10, std::bind(&GoNode::positionCallback, this, std::placeholders::_1));  // 创建位置消息订阅者(回调函数:positionCallback)

                real_angles_sub_ = this->create_subscription<tdt_interface::msg::ReceiveData>(  // 创建角度消息订阅者(回调函数:anglesCallback)
                        angles_topic, 10, std::bind(&GoNode::anglesCallback, this, std::placeholders::_1));

                health_sub_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(  // 创建健康消息订阅者(回调函数:healthCallback)
                        "/game_healths", 10, std::bind(&GoNode::healthCallback, this, std::placeholders::_1));

                std::string camera_topic = (player_id == 1) ? "/camera_image_player_1" : "/camera_image_player_2";  // 根据玩家ID选择相机话题
                camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(  // 创建相机图像订阅者(订阅camera_topic话题,回调函数为cameraCallback,队列大小为10)
                        camera_topic, 10, std::bind(&GoNode::cameraCallback, this, std::placeholders::_1));

                game_time_sub_ = this->create_subscription<std_msgs::msg::Int32>(  // 创建游戏时间订阅者(订阅/game_time话题,回调函数为gameTimeCallback)
                        "/game_time", 10, std::bind(&GoNode::gameTimeCallback, this, std::placeholders::_1));  // 创建游戏时间订阅者(订阅/game_time话题)

        // 初始化时间管理器
                TimeManager::initialize(this->now());  // 初始化时间管理器(设置游戏开始时间)

                std::string speed_topic = (player_id == 1) ? "/target_speed_player_1" : "/target_speed_player_2";  // 根据玩家ID选择速度话题
                std::string target_angles_topic = (player_id == 1) ? "/target_angles_player_1" : "/target_angles_player_2";  // 根据玩家ID选择角度话题

                speed_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(speed_topic, 10);  // 创建速度发布者(发布到speed_topic话题,队列大小为10)
                angles_pub_ = this->create_publisher<tdt_interface::msg::SendData>(target_angles_topic, 10);  // 创建角度发布者(发布到target_angles_topic话题,队列大小为10)

                control_timer_ = this->create_wall_timer(  // 将control_timer_赋值为this->create_wall_timer(
                        std::chrono::milliseconds(5), std::bind(&GoNode::controlLoop, this));  // 创建控制循环定时器(5ms周期)

                shoot_timer_ = this->create_wall_timer(  // 将shoot_timer_赋值为this->create_wall_timer(
                        std::chrono::milliseconds(30), std::bind(&GoNode::shootLoop, this));  // 创建射击循环定时器(30ms周期,与相机帧率接近)

                position_adjust_timer_ = this->create_wall_timer(  // 将position_adjust_timer_赋值为this->create_wall_timer(
                        std::chrono::milliseconds(50), std::bind(&GoNode::positionAdjustLoop, this));  // 创建位置调整循环定时器(50ms周期)

                health_check_timer_ = this->create_wall_timer(  // 将health_check_timer_赋值为this->create_wall_timer(
                        std::chrono::milliseconds(1000), std::bind(&GoNode::healthCheckLoop, this));  // 创建健康检查循环定时器(1秒周期)

                swing_timer_ = this->create_wall_timer(  // 将swing_timer_赋值为this->create_wall_timer(
                                std::chrono::milliseconds(20), std::bind(&GoNode::swingLoop, this));        RCLCPP_INFO(this->get_logger(), "GoNode initialized for Player %d", player_id);  // 创建摆动循环定时器(20ms周期,执行摆动瞄准)
    }

private:
        void loadConfigParameters() {
                // 从ConfigManager读取配置参数，如果配置不可用则使用硬编码默认值
                if (config_manager_ && config_manager_->isLoaded()) {
                        const auto& config = config_manager_->getConfig();
                        
                        // Speed相关
                        max_speed_far_ = 10.0 * config.unit.meter_to_map_unit;
                        max_speed_near_ = 3.0 * config.unit.meter_to_map_unit;
                        max_speed_precise_ = 0.8 * config.unit.meter_to_map_unit;
                        speed_switch_distance_ = 5.0;
                        speed_switch_distance_fast_ = 10.0;
                        precise_mode_distance_ = 1.5;
                        arrival_threshold_fast_ = 2.0;
                        arrival_threshold_normal_ = 0.8;
                        precise_arrival_threshold_ = 0.2;
                        precise_adjust_speed_ = 0.3 * config.unit.meter_to_map_unit;
                        position_hold_threshold_ = config.position_adjust.hold_threshold;
                        
                        // Swing相关
                        swing_range_ = config.swing.range;
                        swing_speed_ = config.swing.speed;
                        
                        // Threshold相关
                        consecutive_miss_threshold_disable_ = config.threshold.consecutive_miss_disable;
                        consecutive_miss_threshold_swing_ = config.threshold.consecutive_miss_swing;
                        hit_threshold_ = config.threshold.hit;
                        
                        // Timeout相关
                        health_history_duration_ = config.timeout.health_history;
                        other_data_duration_ = config.timeout.other_data;
                        move_timeout_ = config.timeout.move;
                        
                        // Filter相关
                        distance_filter_window_ = config.filter.distance_window;
                        yaw_filter_alpha_ = config.filter.yaw_alpha;
                        pitch_filter_alpha_ = config.filter.pitch_alpha;
                        
                        // Camera/Ballistic相关（硬编码，暂不支持配置）
                        camera_fx_ = 554.26;
                        camera_fy_ = 554.26;
                        camera_cx_ = 320.0;
                        camera_cy_ = 240.0;
                        bullet_speed_ = 23.0;
                        gravity_ = 5.0;
                } else {
                        // 使用硬编码默认值
                        max_speed_far_ = 10.0 * 5.0;
                        max_speed_near_ = 3.0 * 5.0;
                        max_speed_precise_ = 0.8 * 5.0;
                        speed_switch_distance_ = 5.0;
                        speed_switch_distance_fast_ = 10.0;
                        precise_mode_distance_ = 1.5;
                        arrival_threshold_fast_ = 2.0;
                        arrival_threshold_normal_ = 0.8;
                        precise_arrival_threshold_ = 0.2;
                        precise_adjust_speed_ = 0.3 * 5.0;
                        position_hold_threshold_ = 0.05;
                        swing_range_ = 6.0;
                        swing_speed_ = 10.0;
                        consecutive_miss_threshold_disable_ = 10;
                        consecutive_miss_threshold_swing_ = 3;
                        hit_threshold_ = -10;
                        health_history_duration_ = 5.0;
                        other_data_duration_ = 10.0;
                        move_timeout_ = 10.0;
                        distance_filter_window_ = 5;
                        yaw_filter_alpha_ = 0.1;
                        pitch_filter_alpha_ = 0.1;
                        camera_fx_ = 554.26;
                        camera_fy_ = 554.26;
                        camera_cx_ = 320.0;
                        camera_cy_ = 240.0;
                        bullet_speed_ = 23.0;
                        gravity_ = 5.0;
                }
        }

        // 验证配置参数的合理性
        bool validateConfig() {
                bool is_valid = true;
                
                // 验证速度参数
                if (max_speed_far_ <= 0 || max_speed_far_ > 100.0) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：max_speed_far %.2f 超出范围 (0, 100]", max_speed_far_);
                        is_valid = false;
                }
                if (max_speed_near_ <= 0 || max_speed_near_ > max_speed_far_) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：max_speed_near %.2f 超出范围 (0, max_speed_far]", max_speed_near_);
                        is_valid = false;
                }
                if (max_speed_precise_ <= 0 || max_speed_precise_ > max_speed_near_) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：max_speed_precise %.2f 超出范围 (0, max_speed_near]", max_speed_precise_);
                        is_valid = false;
                }
                
                // 验证距离参数
                if (speed_switch_distance_ <= 0 || speed_switch_distance_ > 50.0) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：speed_switch_distance %.2f 超出范围 (0, 50]", speed_switch_distance_);
                        is_valid = false;
                }
                if (precise_mode_distance_ <= 0 || precise_mode_distance_ > speed_switch_distance_) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：precise_mode_distance %.2f 超出范围 (0, speed_switch_distance]", precise_mode_distance_);
                        is_valid = false;
                }
                
                // 验证摆动参数
                if (swing_range_ <= 0 || swing_range_ > 45.0) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：swing_range %.2f 超出范围 (0, 45]", swing_range_);
                        is_valid = false;
                }
                if (swing_speed_ <= 0 || swing_speed_ > 60.0) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：swing_speed %.2f 超出范围 (0, 60]", swing_speed_);
                        is_valid = false;
                }
                
                // 验证阈值参数
                if (consecutive_miss_threshold_disable_ <= 0 || consecutive_miss_threshold_disable_ > 100) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：consecutive_miss_threshold_disable %d 超出范围 (0, 100]", consecutive_miss_threshold_disable_);
                        is_valid = false;
                }
                if (consecutive_miss_threshold_swing_ <= 0 || consecutive_miss_threshold_swing_ > consecutive_miss_threshold_disable_) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：consecutive_miss_threshold_swing %d 超出范围 (0, consecutive_miss_threshold_disable]", consecutive_miss_threshold_swing_);
                        is_valid = false;
                }
                
                // 验证超时参数
                if (move_timeout_ <= 0 || move_timeout_ > 300.0) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：move_timeout %.2f 超出范围 (0, 300]", move_timeout_);
                        is_valid = false;
                }
                
                // 验证滤波器参数
                if (distance_filter_window_ <= 0 || distance_filter_window_ > 50) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：distance_filter_window %d 超出范围 (0, 50]", distance_filter_window_);
                        is_valid = false;
                }
                if (yaw_filter_alpha_ <= 0.0 || yaw_filter_alpha_ > 1.0) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：yaw_filter_alpha %.2f 超出范围 (0, 1]", yaw_filter_alpha_);
                        is_valid = false;
                }
                if (pitch_filter_alpha_ <= 0.0 || pitch_filter_alpha_ > 1.0) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：pitch_filter_alpha %.2f 超出范围 (0, 1]", pitch_filter_alpha_);
                        is_valid = false;
                }
                
                // 验证相机参数
                if (camera_fx_ <= 0 || camera_fx_ > 10000.0) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：camera_fx %.2f 超出范围 (0, 10000]", camera_fx_);
                        is_valid = false;
                }
                if (camera_fy_ <= 0 || camera_fy_ > 10000.0) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：camera_fy %.2f 超出范围 (0, 10000]", camera_fy_);
                        is_valid = false;
                }
                
                // 验证弹道参数
                if (bullet_speed_ <= 0 || bullet_speed_ > 1000.0) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：bullet_speed %.2f 超出范围 (0, 1000]", bullet_speed_);
                        is_valid = false;
                }
                if (gravity_ <= 0 || gravity_ > 100.0) {
                        RCLCPP_ERROR(this->get_logger(), "配置错误：gravity %.2f 超出范围 (0, 100]", gravity_);
                        is_valid = false;
                }
                
                if (is_valid) {
                        RCLCPP_INFO(this->get_logger(), "配置验证通过");
                } else {
                        RCLCPP_ERROR(this->get_logger(), "配置验证失败，请检查配置文件");
                }
                
                return is_valid;
        }

        void positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {  // 声明成员函数positionCallback(位置回调函数,接收位置消息,更新机器人的位置数据)
                data_manager_->updatePosition(msg->pose.position.x, msg->pose.position.y, msg->header.stamp);  // 更新位置数据(调用数据管理器更新机器人位置)
    }

        void anglesCallback(const tdt_interface::msg::ReceiveData::SharedPtr msg) {
                current_yaw_ = msg->yaw;  // 将current_yaw_赋值为msg->yaw
                current_pitch_ = msg->pitch;  // 将current_pitch_赋值为msg->pitch
                data_manager_->updateYawPitch(msg->yaw, msg->pitch, msg->timestamp);  // 更新角度数据(调用数据管理器更新云台角度)
    }

        bool checkBaseDestroyed(size_t enemy_internal_index, int enemy_health) {
                // 检测敌方基地是否被摧毁
                size_t base_topic_index = (player_id_ == 1) ? 11 : 5;
                int base_internal_index = healthTopicIndexToInternalIndex(base_topic_index, player_id_);
                
                if (base_internal_index >= 0 && base_internal_index < 6 &&
                        enemy_internal_index == static_cast<size_t>(base_internal_index) &&
                        enemy_health <= 0 &&
                        data_manager_->isEnemyHealthInitialized(base_internal_index)) {
                        RCLCPP_FATAL(this->get_logger(), "基地被摧毁！游戏结束！");
                        rclcpp::shutdown();
                        exit(0);
                        return true;
                }
                return false;
        }

        void healthCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg) {
                auto current_time = TimeManager::getCurrentTime();
        
                // 更新自己血量
                size_t self_index = (player_id_ == 1) ? 0 : 6;
                if (self_index < msg->data.size()) {
                        data_manager_->updateSelfHealth(msg->data[self_index], current_time);
        }

                // 更新6个敌人血量
                for (size_t i = 0; i < 6; i++) {
                        size_t enemy_index = (player_id_ == 1) ? (6 + i) : i;
                        if (enemy_index < msg->data.size()) {
                                data_manager_->updateEnemyHealth(i, msg->data[enemy_index], current_time);
                                checkBaseDestroyed(i, msg->data[enemy_index]);
            }
        }
    }

        void cameraCallback(const sensor_msgs::msg::Image::SharedPtr msg) {  // 声明成员函数cameraCallback(相机回调函数,接收图像消息,进行目标检测)
                try {  // 开始异常捕获(准备处理可能出现的异常)
                        if (!msg) {
                                RCLCPP_ERROR(this->get_logger(), "相机消息为空");
                                return;
                        }
                        
                        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");  // 将cv_ptr赋值为cv_bridge::toCvCopy(msg, "bgr8")
                        
                        // 验证图像尺寸
                        if (cv_ptr->image.empty()) {
                                RCLCPP_ERROR(this->get_logger(), "图像为空");
                                return;
                        }
                        
                        if (cv_ptr->image.cols <= 0 || cv_ptr->image.rows <= 0) {
                                RCLCPP_ERROR(this->get_logger(), "图像尺寸无效: %dx%d", 
                                            cv_ptr->image.cols, cv_ptr->image.rows);
                                return;
                        }
                        
                        centernet_detector_->updateFrame(cv_ptr->image);  // 更新图像帧(调用检测器进行目标检测)
                        
                } catch (cv_bridge::Exception& e) {
                        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());  // 输出错误日志(图像转换异常)
                } catch (const std::exception& e) {
                        RCLCPP_ERROR(this->get_logger(), "相机回调异常: %s", e.what());  // 输出错误日志(其他异常)
                } catch (...) {
                        RCLCPP_ERROR(this->get_logger(), "相机回调未知异常");  // 输出错误日志(未知异常)
        }
    }

        void gameTimeCallback(const std_msgs::msg::Int32::SharedPtr msg) {
                TimeManager::updateGameTime(msg->data);  // 更新游戏时间(更新当前游戏时长)
    }

        void controlLoop() {
                auto current_time = TimeManager::getCurrentTime();
                
                // 看门狗检查
                {
                        std::lock_guard<std::mutex> lock(watchdog_mutex_);
                        control_loop_last_update_ = current_time;
                }
                
                size_t waypoint_idx = path_navigator_->getCurrentWaypointIndex();  // 将waypoint_idx赋值为path_navigator_->getCurrentWaypointIndex()

                if (waypoint_idx >= waypoints_->size()) return;

                const Waypoint& target = (*waypoints_)[waypoint_idx];
                auto [current_x, current_y] = data_manager_->getCurrentPosition();  // 将current_y]赋值为data_manager_->getCurrentPosition()

                double dx = target.x - current_x;  // 将dx赋值为target.x - current_x
                double dy = target.y - current_y;  // 将dy赋值为target.y - current_y
                double distance = std::sqrt(dx * dx + dy * dy);  // 将distance赋值为std::sqrt(dx * dx + dy * dy)

          // 判断是否超过指定时间(用于超时检测)
                if (path_navigator_->isTimeout(current_time) && state_ == MOVING) {  // 判断是否超时
                        path_navigator_->incrementRetryCount(current_time);  // 增加重试计数器(记录卡住后重试的次数)
            
                        if (path_navigator_->isMaxRetriesExceeded()) {  // 如果条件满足,执行对应操作
                                RCLCPP_INFO(this->get_logger(), "P%zu: 3次尝试失败，跳过该点位", waypoint_idx + 1);  // 输出信息日志
                                path_navigator_->skipCurrentPoint(current_time);  // 跳过当前路径点(移动到下一个路径点)
                        } else {  // 否则(敌人血量大于0)
                                RCLCPP_INFO(this->get_logger(), "P%zu: 超时，返回上一位点（重试%d/3）",  // 输出信息日志
                                                    waypoint_idx + 1, path_navigator_->getRetryCount());
                                path_navigator_->returnToPrevious();  // 返回上一个路径点(用于P7-P11循环)
            }
                        return;  // 返回函数(退出函数,不返回值)
        }

        // 检查SHOOTING状态下是否被推得太远（等待5秒后再判定）
                if (state_ == SHOOTING && distance > arrival_threshold_normal_ * 2) {  // 如果条件满足,执行对应操作
                        if (TimeManager::isTimeout(shooting_start_time_, current_time, 5.0)) {
                                RCLCPP_INFO(this->get_logger(), "P%zu: SHOOTING状态下被推得太远（%.2f），切换到MOVING状态重新导航",  // 输出信息日志
                                                    waypoint_idx + 1, distance);
                                shoot_controller_->stopShooting();  // 调用stopShooting方法(停止射击,禁用自动瞄准)
                                state_ = MOVING;  // 将state_赋值为MOVING
                                path_navigator_->resetTimeoutTimer(current_time);  // 重置超时计时器
                path_navigator_->resetRetryCount();  // 重置重试计数器
                                return;  // 返回函数(退出函数,不返回值)
            }
        }

        // 检查是否到达目标点
        // 根据点位类型选择不同的到达精度
                double arrival_threshold = arrival_threshold_normal_;  // 将arrival_threshold赋值为arrival_threshold_normal_
                double precise_arrival_threshold = precise_arrival_threshold_;  // 将precise_arrival_threshold赋值为precise_arrival_threshold_
        
                if (waypoint_idx <= 2) {  // 如果路径点索引小于等于2(前3个路径点,快速到达)
            // P1-P3：快速到达，低精度要求
                        arrival_threshold = arrival_threshold_fast_;  // 将arrival_threshold赋值为arrival_threshold_fast_
            precise_arrival_threshold = arrival_threshold_fast_;  // 设置快速到达的阈值(ARRIVAL_THRESHOLD_FAST=0.3米)
        }
        
                if (distance <= arrival_threshold) {  // 将<赋值为arrival_threshold) {
                        if (state_ == MOVING) {  // 如果条件满足,执行对应操作
                                                                path_navigator_->resetTimeoutTimer(current_time);  // 重置超时计时器
                                                                is_precise_adjusting_ = false;  // 将is_precise_adjusting_设置为false
            
                                                                if (distance > precise_arrival_threshold) {  // 如果条件满足,执行对应操作
                                        is_precise_adjusting_ = true;  // 将is_precise_adjusting_设置为true
                                        preciseAdjustment(dx, dy, distance);  // 执行精确位置调整(根据距离调整移动速度)
                    return;  // 精确调整后return，避免重复发送速度命令
                }

                                if (target.has_shoot_task) {  // 如果条件满足,执行对应操作
                    // 检查是否在被攻击状态下到达新的射击任务点位
                                        if (is_under_attack_ && !attack_waypoint_stack_.empty() && waypoint_idx != attack_waypoint_stack_.top()) {  // 如果条件满足,执行对应操作
                                                int self_health_change = data_manager_->getSelfHealthChange();  // 将self_health_change赋值为data_manager_->getSelfHealthChange()
                        // 只有当自身血量仍然减少时才返回上一点位
                                                if (self_health_change < 0) {
                                                        RCLCPP_INFO(this->get_logger(), "P%zu: 被攻击状态下到达，自身血量仍在减少(%d)，返回上一点位P%zu",  // 输出信息日志
                                                                                                                                    waypoint_idx + 1, self_health_change, attack_waypoint_stack_.top() + 1);
                            // 返回上一点位继续任务
                                                        moveToWaypoint(attack_waypoint_stack_.top(), current_time);  // 移动到上一个射击路径点
                                                        return;  // 返回函数(退出函数,不返回值)
                                                } else {  // 否则(后面的路径点,精确到达)
                            // 自身血量不再减少，清除被攻击状态
                                                        is_under_attack_ = false;
                                                        attack_waypoint_stack_.pop();  // 弹出当前点
                                                        RCLCPP_INFO(this->get_logger(), "P%zu: 自身血量不再减少，清除被攻击状态", waypoint_idx + 1);  // 输出信息日志
                        }
                    }
                    
                    RCLCPP_INFO(this->get_logger(), "P%zu: 状态转换 MOVING -> SHOOTING", waypoint_idx + 1);  // 添加状态转换日志
                                        state_ = SHOOTING;  // 将state_赋值为SHOOTING
                                        shooting_start_time_ = current_time;  // 将shooting_start_time_赋值为current_time
                                        int enemy_topic_index = getEnemyIndex(waypoint_idx);  // 将enemy_topic_index赋值为getEnemyIndex(waypoint_idx)
                                        int enemy_internal_index = healthTopicIndexToInternalIndex(enemy_topic_index, player_id_);
                                        int initial_enemy_health = 0;  // 将initial_enemy_health初始化为0
                                        if (enemy_internal_index >= 0 && enemy_internal_index < 6) {  // 将>赋值为0 && enemy_internal_index < 6) {
                                                initial_enemy_health = data_manager_->getInitialEnemyHealth(enemy_internal_index);  // 将initial_enemy_health赋值为data_manager_->getInitialEnemyHealth(enemy_internal_index)
                    }
                                        shoot_controller_->startShooting(initial_enemy_health, current_time);  // 开始射击
                                        target_yaw_ = target.yaw;  // 将target_yaw_赋值为target.yaw
                                        target_pitch_ = target.pitch;  // 将target_pitch_赋值为target.pitch
                                        current_yaw_ = target.yaw;  // 将current_yaw_赋值为target.yaw
                                        current_pitch_ = target.pitch;  // 将current_pitch_赋值为target.pitch
                                        RCLCPP_INFO(this->get_logger(), "到达P%zu，开始射击任务", waypoint_idx + 1);  // 输出信息日志
                                } else {  // 否则(未到达目标点)
                    // 到达普通点，准备前往下一点
                    // 先保存当前索引，避免moveToNext后索引变化
                                        size_t current_idx = path_navigator_->getCurrentWaypointIndex();  // 将current_idx赋值为path_navigator_->getCurrentWaypointIndex()
                                        size_t next_idx = current_idx + 1;  // 将next_idx赋值为current_idx + 1
                    
                                        if (next_idx < waypoints_->size()) {
                        // 调整到下一个点位的yaw和pitch（不管是否有射击任务）
                                                target_yaw_ = (*waypoints_)[next_idx].yaw;
                                                target_pitch_ = (*waypoints_)[next_idx].pitch;
                                                current_yaw_ = target_yaw_;  // 将current_yaw_赋值为target_yaw_
                                                current_pitch_ = target_pitch_;  // 将current_pitch_赋值为target_pitch_
                        
                        // 发布角度命令到游戏
                                                auto angle_msg = tdt_interface::msg::SendData();  // 将angle_msg赋值为tdt_interface::msg::SendData()
                                                angle_msg.yaw = current_yaw_;  // 将angle_msg.yaw赋值为current_yaw_
                                                angle_msg.pitch = current_pitch_;  // 将angle_msg.pitch赋值为current_pitch_
                                                angle_msg.if_shoot = false;  // 将angle_msg.if_shoot设置为false
                                                angles_pub_->publish(angle_msg);  // 声明角度发布者(发送目标角度和射击指令)
                        
                                                RCLCPP_INFO(this->get_logger(), "离开P%zu，调整yaw/pitch到P%zu（%.1f, %.1f）",  // 输出信息日志
                                                                    current_idx + 1, next_idx + 1, target_yaw_, target_pitch_);
                    }
                                        path_navigator_->moveToNext(current_time);  // 移动到下一个路径点
                }
            }
                } else if (state_ == MOVING) {  // 如果条件满足,执行对应操作
                        moveTowardsTarget(dx, dy, distance, target, waypoint_idx);  // 向目标路径点移动
        }
    }

        void moveTowardsTarget(double dx, double dy, double distance, const Waypoint& target, size_t waypoint_idx) {
                double direction_x = dx / distance;  // 将direction_x赋值为dx / distance
                double direction_y = dy / distance;  // 将direction_y赋值为dy / distance

        // 根据点位类型选择不同的速度控制策略
                size_t current_idx = path_navigator_->getCurrentWaypointIndex();
                double speed_switch_distance = speed_switch_distance_;  // 将speed_switch_distance赋值为speed_switch_distance_
        
                if (current_idx <= 2) {  // 如果当前路径点索引小于等于2(前3个路径点,快速到达)
            // P1-P3：使用更大的切换距离，全程高速
                        speed_switch_distance = speed_switch_distance_fast_;  // 将speed_switch_distance赋值为speed_switch_distance_fast_
        }

                double current_max_speed;  // 声明当前最大速度变量
                if (distance > speed_switch_distance) {  // 如果条件满足,执行对应操作
                        current_max_speed = max_speed_far_;  // 将current_max_speed赋值为max_speed_far_
                } else if (distance > precise_mode_distance_) {  // 如果条件满足,执行对应操作
                        current_max_speed = max_speed_near_;  // 将current_max_speed赋值为max_speed_near_
                } else {  // 否则分支
                        double speed_ratio = distance / precise_mode_distance_;  // 将speed_ratio赋值为distance / precise_mode_distance_
                        current_max_speed = max_speed_precise_ + (max_speed_near_ - max_speed_precise_) * speed_ratio;  // 将current_max_speed赋值为max_speed_precise_ + (max_speed_near_ - max_speed_precise_) * speed_ratio
        }

                auto message = geometry_msgs::msg::TwistStamped();  // 将message赋值为geometry_msgs::msg::TwistStamped()
                message.header.stamp = TimeManager::getCurrentTime();  // 设置消息时间戳为当前游戏时间(用于ROS2消息同步)
                message.header.frame_id = "base_link";  // 设置消息坐标系为base_link(机器人底盘坐标系)
                message.twist.linear.x = direction_x * current_max_speed;  // 将message.twist.linear.x赋值为direction_x * current_max_speed
                message.twist.linear.y = direction_y * current_max_speed;  // 将message.twist.linear.y赋值为direction_y * current_max_speed
                speed_pub_->publish(message);  // 声明速度发布者(发送机器人速度指令)
    }

        void healthCheckLoop() {
                // 定期输出性能监控报告（每10秒）
                static int health_check_count_ = 0;
                if (++health_check_count_ >= 10) {  // 10秒 = 10次调用（1秒/次）
                        health_check_count_ = 0;
                        if (performance_monitor_) {
                                performance_monitor_->reportAll();
                        }
                }
        
                if (!shoot_controller_->isShooting()) return;  // 如果条件满足,执行对应操作

                auto current_time = TimeManager::getCurrentTime();
                size_t waypoint_idx = path_navigator_->getCurrentWaypointIndex();

                if (waypoint_idx >= waypoints_->size() || !(*waypoints_)[waypoint_idx].has_shoot_task) {
                        return;  // 返回函数(退出函数,不返回值)
        }

                int health_change_threshold = getHealthChangeThreshold(waypoint_idx);  // 将health_change_threshold赋值为getHealthChangeThreshold(waypoint_idx)
                int enemy_topic_index = getEnemyIndex(waypoint_idx);
                int enemy_internal_index = healthTopicIndexToInternalIndex(enemy_topic_index, player_id_);
        
                int current_enemy_health = 0;  // 将current_enemy_health初始化为0
                int initial_enemy_health = 0;
                int health_change = 0;  // 将health_change初始化为0
                bool enemy_health_valid = false;  // 将enemy_health_valid设置为false
        
                if (enemy_internal_index >= 0 && enemy_internal_index < 6) {
                        current_enemy_health = data_manager_->getCurrentEnemyHealth(enemy_internal_index);  // 将current_enemy_health赋值为data_manager_->getCurrentEnemyHealth(enemy_internal_index)
                        initial_enemy_health = data_manager_->getInitialEnemyHealth(enemy_internal_index);
                        health_change = data_manager_->getEnemyHealthChange(enemy_internal_index);  // 将health_change赋值为data_manager_->getEnemyHealthChange(enemy_internal_index)
            
                        if (data_manager_->isEnemyHealthInitialized(enemy_internal_index)) {  // 如果条件满足,执行对应操作
                                enemy_health_valid = true;  // 将enemy_health_valid设置为true
            }
        }

                RCLCPP_INFO(this->get_logger(), "P%zu: 血量变化=%d, 阈值=%d, 连续未达标=%d",  // 输出信息日志
                                    waypoint_idx + 1, health_change, health_change_threshold,
                                    shoot_controller_->getConsecutiveMissCountForSwing());

                bool health_changed = shoot_controller_->checkHealthChange(health_change, health_change_threshold, current_time);  // 将health_changed赋值为shoot_controller_->checkHealthChange(health_change, health_change_threshold, current_time)

                if (health_changed) {  // 如果条件满足,执行对应操作
                        RCLCPP_INFO(this->get_logger(), "P%zu: 血量变化达到明显标准，重置计数", waypoint_idx + 1);  // 输出信息日志
        }

                bool should_disable = shoot_controller_->shouldDisableAutoAim(current_time);  // 将should_disable赋值为shoot_controller_->shouldDisableAutoAim(current_time)
                if (should_disable) {  // 如果条件满足,执行对应操作
                        shoot_controller_->disableAutoAim();  // 禁用自动瞄准
                        RCLCPP_INFO(this->get_logger(), "P%zu: 禁用自瞄，切换到硬编码", waypoint_idx + 1);  // 输出信息日志
        }

                bool should_swing = shoot_controller_->shouldStartSwinging(current_time);  // 将should_swing赋值为shoot_controller_->shouldStartSwinging(current_time)
                if (should_swing && !shoot_controller_->isSwinging()) {  // 如果条件满足,执行对应操作
                        shoot_controller_->startSwinging(current_yaw_, current_time);  // 开始摆动瞄准
                        RCLCPP_INFO(this->get_logger(), "P%zu: 连续未达标触发摆动", waypoint_idx + 1);  // 输出信息日志
        }

                if (shoot_controller_->isSwinging() && health_changed) {  // 如果条件满足,执行对应操作
                        shoot_controller_->stopSwinging(current_time);  // 停止摆动瞄准
                        auto [prev_yaw, prev_pitch] = data_manager_->getHistoricalYawPitch(current_time);  // 将prev_pitch]赋值为data_manager_->getHistoricalYawPitch(current_time)
                        current_yaw_ = prev_yaw;  // 将current_yaw_赋值为prev_yaw
                        RCLCPP_INFO(this->get_logger(), "P%zu: 检测到血量变化，停止摆动，调整yaw到前一秒数值", waypoint_idx + 1);  // 输出信息日志
        }

                int self_health_change = data_manager_->getSelfHealthChange();
                if (self_health_change < 0) {
                        if (enemy_health_valid && initial_enemy_health > 0) {  // 如果条件满足,执行对应操作
                                RCLCPP_INFO(this->get_logger(), "P%zu: 检测到自身血量减少%d，被攻击", waypoint_idx + 1, self_health_change);  // 输出信息日志
                                handleSelfUnderAttack(waypoint_idx, current_enemy_health, initial_enemy_health);  // 处理自己被攻击的情况
            }
        }

                if (enemy_health_valid && current_enemy_health <= 0) {  // 如果敌人血量有效且当前血量小于等于0(敌人已被击杀,处理击杀逻辑)
                        RCLCPP_INFO(this->get_logger(), "P%zu: 敌人击杀，状态转换 SHOOTING -> MOVING", waypoint_idx + 1);  // 添加状态转换日志
                        shoot_controller_->stopShooting();  // 调用stopShooting方法(停止射击,禁用自动瞄准)
                        state_ = MOVING;
                        is_under_attack_ = false;
                        while (!attack_waypoint_stack_.empty()) {  // 清空被攻击点位栈
                                attack_waypoint_stack_.pop();
                        }
                        if (path_navigator_->isReturning()) {  // 如果条件满足,执行对应操作
                                path_navigator_->cancelReturn();  // 取消返回
            }
            
            // 调整到下一个点位的yaw和pitch
                        size_t current_idx = path_navigator_->getCurrentWaypointIndex();
                        size_t next_idx = current_idx + 1;
                        if (next_idx < waypoints_->size()) {
                                target_yaw_ = (*waypoints_)[next_idx].yaw;
                                target_pitch_ = (*waypoints_)[next_idx].pitch;
                                current_yaw_ = target_yaw_;
                                current_pitch_ = target_pitch_;
                
                // 发布角度命令到游戏
                                auto angle_msg = tdt_interface::msg::SendData();
                                angle_msg.yaw = current_yaw_;
                                angle_msg.pitch = current_pitch_;
                                angle_msg.if_shoot = false;
                                angles_pub_->publish(angle_msg);  // 声明角度发布者(发送目标角度和射击指令)
                
                                RCLCPP_INFO(this->get_logger(), "P%zu: 目标被摧毁，调整yaw/pitch到P%zu（%.1f, %.1f）",  // 输出信息日志
                                                    current_idx + 1, next_idx + 1, target_yaw_, target_pitch_);
            }
            
                        path_navigator_->moveToNext(current_time);  // 移动到下一个路径点
                        RCLCPP_INFO(this->get_logger(), "P%zu: 目标被摧毁，前往下一点", waypoint_idx + 1);  // 输出信息日志
        }
    }

        void swingLoop() {
        // 只在射击状态下执行
                if (state_ != SHOOTING) return;  // 如果条件满足,执行对应操作
        
                if (!shoot_controller_->isShooting()) return;  // 如果条件满足,执行对应操作

                size_t waypoint_idx = path_navigator_->getCurrentWaypointIndex();
                if (waypoint_idx >= waypoints_->size()) return;

                const Waypoint& target = (*waypoints_)[waypoint_idx];

            // 条件判断
                if (shoot_controller_->isSwinging()) {  // 如果条件满足,执行对应操作
                        auto current_time = TimeManager::getCurrentTime();
            double swing_yaw = shoot_controller_->getSwingYaw(0.02);  // 20ms = 0.02s
            
                        current_yaw_ = target.yaw + swing_yaw;  // 将current_yaw_赋值为target.yaw + swing_yaw
                        current_pitch_ = target.pitch;
                } else {  // 否则分支
            // 不在摆动模式时，使用shootLoop中已经计算好的current_yaw_和current_pitch_
            // 或者使用硬编码角度
                        if (!shoot_controller_->isAutoAimEnabled()) {  // 如果条件满足,执行对应操作
                                current_yaw_ = target.yaw;
                                current_pitch_ = target.pitch;
            }
        }

        // 统一在这里发布射击命令
                RCLCPP_DEBUG(this->get_logger(), "P%zu: 发布射击命令", waypoint_idx + 1);  // 添加射击日志
                publishShootCommand(waypoint_idx);  // 发送射击指令给机器人(执行攻击)
    }

        void shootLoop() {
                if (!shoot_controller_->isShooting()) return;  // 如果条件满足,执行对应操作

                auto current_time = TimeManager::getCurrentTime();
                
                // 看门狗检查
                {
                        std::lock_guard<std::mutex> lock(watchdog_mutex_);
                        shoot_loop_last_update_ = current_time;
                }
                
                size_t waypoint_idx = path_navigator_->getCurrentWaypointIndex();

                if (waypoint_idx >= waypoints_->size() || !(*waypoints_)[waypoint_idx].has_shoot_task) {
                        shoot_controller_->stopShooting();  // 调用stopShooting方法(停止射击,禁用自动瞄准)
                        return;  // 返回函数(退出函数,不返回值)
        }

                const Waypoint& target = (*waypoints_)[waypoint_idx];
        
                if (shoot_controller_->isAutoAimEnabled()) {  // 如果自动瞄准已启用(检查射击控制器的自动瞄准标志,true表示可以进行自动瞄准)
                        performance_monitor_->startTimer("inference");  // 开始计时推理延迟
                        auto armors = centernet_detector_->detectArmors();  // 将armors赋值为centernet_detector_->detectArmors()
                        performance_monitor_->endTimer("inference");  // 结束计时推理延迟
                        
                        // 添加检测日志
                        static int detect_debug_count = 0;
                        if (++detect_debug_count % 30 == 0) {  // 每30帧输出一次
                                RCLCPP_DEBUG(this->get_logger(), "检测到 %zu 个目标", armors.size());
                        }
                        
                        if (!armors.empty()) {  // 检查数组是否为空
                                detection_fail_count_ = 0;  // 重置检测失败计数器
                // 实时更新目标中心（不再只锁定一次）
                                locked_target_center_ = armors[0].center;  // 将locked_target_center_赋值为armors[0].center
                                target_locked_ = true;  // 将target_locked_设置为true
                
                // 更新目标位置历史
                                centernet_detector_->updateTargetHistory(locked_target_center_);  // 更新目标检测器的目标历史记录(传入锁定的目标中心点坐标,用于平滑追踪和预测目标位置)
                
                // PNP求解获取实时距离和位置
                                auto [distance_meters, position] = centernet_detector_->solvePnPDistance(armors[0].armorBbox);  // 将position]赋值为centernet_detector_->solvePnPDistance(armors[0].armorBbox)
                
                // 使用运动预测的目标中心(时间参数0.01秒,与path_debug.cpp一致)
                                cv::Point2f predicted_center = centernet_detector_->predictTargetPosition(0.01);
                                
                                // 调试输出: 查看为什么突然摆动
                                static int debug_frame = 0;
                                if (debug_frame++ % 30 == 0) {
                                    double dx = predicted_center.x - 320.0;  // 使用默认的camera_cx值
                                    RCLCPP_DEBUG(this->get_logger(), "识别到目标: predicted_center=(%.2f,%.2f) dx=%.2f", predicted_center.x, predicted_center.y, dx);
                                }
                                
                                auto [yaw_offset, pitch_offset] = centernet_detector_->imageToGimbalAngles(predicted_center);  // 将pitch_offset]赋值为centernet_detector_->imageToGimbalAngles(predicted_center)
                                
                                // 调试输出
                                if (debug_frame % 30 == 1) {
                                    RCLCPP_DEBUG(this->get_logger(), "yaw_offset=%.2f°, current_yaw_=%.2f°", yaw_offset, current_yaw_);
                                }
                                
                                auto [final_yaw, final_pitch] = centernet_detector_->calculateBallisticCompensation(  // 将final_pitch]赋值为centernet_detector_->calculateBallisticCompensation(
                                        yaw_offset, pitch_offset, distance_meters, current_yaw_, current_pitch_);  // 计算带有偏移的射击角度,内部已包含滤波
                                
                                if (debug_frame % 30 == 2) {
                                    RCLCPP_DEBUG(this->get_logger(), "final_yaw=%.2f°, final_pitch=%.2f°", final_yaw, final_pitch);
                                }
                                
                                current_yaw_ = final_yaw;  // 将current_yaw_赋值为final_yaw(滤波后的角度)
                                current_pitch_ = final_pitch;  // 将current_pitch_赋值为final_pitch(滤波后的角度)
                        } else {  // 否则分支
                                target_locked_ = false;  // 将target_locked_设置为false
                                detection_fail_count_++;  // 检测失败计数器加1
                                
                                // 降级策略：连续检测失败超过阈值时，使用硬编码角度
                                if (detection_fail_count_ >= MAX_DETECTION_FAIL_COUNT) {
                                        RCLCPP_WARN(this->get_logger(), "连续检测失败%d次，启用降级策略使用硬编码角度", detection_fail_count_);
                                        current_yaw_ = target.yaw;  // 使用硬编码角度
                                        current_pitch_ = target.pitch;
                                        detection_fail_count_ = 0;  // 重置计数器
            }
                }
                } else {  // 否则分支
                        current_yaw_ = target.yaw;
                        current_pitch_ = target.pitch;
        }

        // 不在这里调用publishShootCommand，让swingLoop负责发布
                
                // 更新看门狗
                shoot_loop_last_update_ = current_time;
    }

        void handleSelfUnderAttack(size_t waypoint_idx, int current_enemy_health, int initial_enemy_health) {
                auto current_time = TimeManager::getCurrentTime();
        
        // P7-P11（索引6-10）：只要被攻击就立即移动，不管目标血量
                if (waypoint_idx >= 6 && waypoint_idx <= 10) {  // 将>赋值为6 && waypoint_idx <
                        size_t next_shoot_idx = waypoint_idx + 1;  // 将next_shoot_idx赋值为waypoint_idx + 1
                        if (next_shoot_idx > 10) {  // 如果条件满足,执行对应操作
                next_shoot_idx = 6;  // P11的下一个是P7
            }
            
                        RCLCPP_INFO(this->get_logger(), "P%zu: 被攻击，立即前往下一点位P%zu",  // 输出信息日志
                                            waypoint_idx + 1, next_shoot_idx + 1);
            
              // 退出射击状态并停止自动瞄准
                        RCLCPP_INFO(this->get_logger(), "P%zu: 被攻击，状态转换 SHOOTING -> MOVING", waypoint_idx + 1);  // 添加状态转换日志
                        shoot_controller_->stopShooting();  // 调用stopShooting方法(停止射击,禁用自动瞄准)
                        state_ = MOVING;
            
                        moveToWaypoint(next_shoot_idx, current_time);  // 移动到下一个射击路径点
                        is_under_attack_ = false;
                        return;  // 返回函数(退出函数,不返回值)
        }
        
        // P4-P6：检查目标血量是否有效
                if (initial_enemy_health <= 0) {  // 如果敌人初始血量无效(小于等于0表示数据错误,跳过该敌人)
                        return;  // 返回函数(退出函数,不返回值)
        }
        
                double health_ratio = (double)current_enemy_health / initial_enemy_health;  // 将health_ratio赋值为(double)current_enemy_health / initial_enemy_health
        
        // 若被攻击单位血量不足初始20%，则仍继续打击任务
                if (health_ratio <= 0.2) {  // 将<赋值为0.2) {
                        RCLCPP_INFO(this->get_logger(), "P%zu: 被攻击，但目标血量不足20%%(%.1f%%)，继续打击任务",  // 输出信息日志
                                            waypoint_idx + 1, health_ratio * 100);
                        return;  // 返回函数(退出函数,不返回值)
        }
        
        // 若被攻击单位血量大于初始20%，则前往下一有射击任务点位执行射击任务
                size_t next_shoot_idx = findNextShootTaskIndex(waypoint_idx);  // 将next_shoot_idx赋值为findNextShootTaskIndex(waypoint_idx)
        
                if (next_shoot_idx == waypoint_idx) {  // 如果条件满足,执行对应操作
            // 已经是最后一个射击任务点（P11），下一个是P7
                        next_shoot_idx = 6;  // 将next_shoot_idx赋值为6
        }
        
                RCLCPP_INFO(this->get_logger(), "P%zu: 被攻击，目标血量%.1f%%，前往下一射击任务点P%zu",  // 输出信息日志
                                    waypoint_idx + 1, health_ratio * 100, next_shoot_idx + 1);
        
        // 保存当前射击任务点位索引到栈中（避免连续被攻击时丢失历史）
                if (attack_waypoint_stack_.empty() || attack_waypoint_stack_.top() != waypoint_idx) {
                        attack_waypoint_stack_.push(waypoint_idx);
                }
                is_under_attack_ = true;  // 将is_under_attack_设置为true
        
        // 前往下一有射击任务点位
                moveToWaypoint(next_shoot_idx, current_time);
    }
    
    // 查找下一个有射击任务的点位索引
        size_t findNextShootTaskIndex(size_t current_idx) {
                size_t next_idx = current_idx + 1;
        
            // 条件判断
                if (next_idx >= waypoints_->size()) {
                        next_idx = 6;  // 将next_idx赋值为6
        }
        
        // 查找下一个有射击任务的点位
                while (next_idx < waypoints_->size() && !(*waypoints_)[next_idx].has_shoot_task) {
                        next_idx++;  // 自增
                        if (next_idx >= waypoints_->size()) {
                                next_idx = 6;
            }
        }
        
                return next_idx;  // 返回结果
    }
    
    // 移动到指定点位
        void moveToWaypoint(size_t waypoint_idx, rclcpp::Time current_time) {
                RCLCPP_INFO(this->get_logger(), "开始导航到 P%zu", waypoint_idx + 1);  // 添加导航日志
                size_t current_target = path_navigator_->getCurrentWaypointIndex();  // 将current_target赋值为path_navigator_->getCurrentWaypointIndex()
        
        // 特殊处理：P7-P11循环（索引6-10），支持从P11回到P7
                if (waypoint_idx == 6 && current_target == 10) {  // 如果条件满足,执行对应操作
            path_navigator_->moveToNext(current_time);  // 移动到下一个路径点
                        return;  // 返回函数(退出函数,不返回值)
        }
        
        // 设置目标索引并开始移动
                while (current_target < waypoint_idx) {  // 循环追赶当前路径点
                        path_navigator_->moveToNext(current_time);  // 移动到下一个路径点
                        current_target = path_navigator_->getCurrentWaypointIndex();
            
            // 防止死循环：如果索引没有变化，强制前进
                        if (path_navigator_->isReturning()) {  // 如果条件满足,执行对应操作
                                path_navigator_->cancelReturn();  // 取消返回
            }
        }
    }

        // 检查所有定时器是否正常工作（看门狗机制）
        void checkWatchdogs(rclcpp::Time current_time) {
                static int watchdog_check_count = 0;
                if (++watchdog_check_count < 100) return;  // 每100次检查一次
                watchdog_check_count = 0;
                
                std::lock_guard<std::mutex> lock(watchdog_mutex_);
                
                auto control_elapsed = TimeManager::timeSince(control_loop_last_update_, current_time);
                auto shoot_elapsed = TimeManager::timeSince(shoot_loop_last_update_, current_time);
                
                if (control_elapsed > 0.1) {  // 控制循环超过100ms未更新
                        RCLCPP_WARN(this->get_logger(), "控制循环可能卡死！最后更新: %.2f秒前", control_elapsed);
                }
                
                if (shoot_elapsed > 0.1) {  // 射击循环超过100ms未更新
                        RCLCPP_WARN(this->get_logger(), "射击循环可能卡死！最后更新: %.2f秒前", shoot_elapsed);
                }
        }

        void publishShootCommand(size_t waypoint_idx) {
                auto message = tdt_interface::msg::SendData();  // 将message赋值为tdt_interface::msg::SendData()

                if (shoot_controller_->isSwinging()) {  // 如果条件满足,执行对应操作
            // 使用shootLoop中已经更新好的current_yaw_，不再调用getSwingYaw
                        message.yaw = current_yaw_;  // 云台偏航角(左右转动角度,度)
                        message.pitch = current_pitch_;  // 云台俯仰角(上下转动角度,度)
                } else if (shoot_controller_->isAutoAimEnabled()) {  // 如果条件满足,执行对应操作
            // 使用shootLoop中已经计算好的current_yaw_和current_pitch_
                        message.yaw = current_yaw_;
                        message.pitch = current_pitch_;
                } else {  // 否则分支
                        message.yaw = (*waypoints_)[waypoint_idx].yaw;
                        message.pitch = (*waypoints_)[waypoint_idx].pitch;
        }
        
                message.if_shoot = true;  // 将message.if_shoot设置为true
                angles_pub_->publish(message);  // 声明角度发布者(发送目标角度和射击指令)
    }

        void positionAdjustLoop() {
                        if (state_ != SHOOTING) return;  // 如果条件满足,执行对应操作
            if (is_precise_adjusting_) return;  // 精确调整时不进行位置保持
    
                        auto current_time = TimeManager::getCurrentTime();
                        size_t waypoint_idx = path_navigator_->getCurrentWaypointIndex();
    
                        if (waypoint_idx >= waypoints_->size()) return;
    
                        const Waypoint& target = (*waypoints_)[waypoint_idx];
                        auto [current_x, current_y] = data_manager_->getCurrentPosition();
    
                        double dx = target.x - current_x;
                        double dy = target.y - current_y;
                        double distance = std::sqrt(dx * dx + dy * dy);
    
                // 条件判断
                        if (distance > position_hold_threshold_) {  // 如果条件满足,执行对应操作
                                double direction_x = dx / distance;
                                double direction_y = dy / distance;
    
                                auto message = geometry_msgs::msg::TwistStamped();
                                message.header.stamp = current_time;
                                message.header.frame_id = "base_link";
                                message.twist.linear.x = direction_x * precise_adjust_speed_;  // 将message.twist.linear.x赋值为direction_x * precise_adjust_speed_
                                message.twist.linear.y = direction_y * precise_adjust_speed_;  // 将message.twist.linear.y赋值为direction_y * precise_adjust_speed_
                                speed_pub_->publish(message);  // 声明速度发布者(发送机器人速度指令)
            }
        }
        void preciseAdjustment(double dx, double dy, double distance) {
                double direction_x = dx / distance;
                double direction_y = dy / distance;
        
        // 根据距离减速：越接近目标，速度越慢
                double speed_ratio = std::min(1.0, distance / precise_arrival_threshold_);  // 将speed_ratio赋值为std::min(1.0, distance / precise_arrival_threshold_)
                double adjust_speed = precise_adjust_speed_ * speed_ratio;  // 将adjust_speed赋值为precise_adjust_speed_ * speed_ratio

                auto message = geometry_msgs::msg::TwistStamped();
                message.header.stamp = TimeManager::getCurrentTime();
                message.header.frame_id = "base_link";
                message.twist.linear.x = direction_x * adjust_speed;  // 将message.twist.linear.x赋值为direction_x * adjust_speed
                message.twist.linear.y = direction_y * adjust_speed;  // 将message.twist.linear.y赋值为direction_y * adjust_speed
                speed_pub_->publish(message);  // 声明速度发布者(发送机器人速度指令)
    }

        int getHealthChangeThreshold(size_t waypoint_idx) {
                // 优先使用配置文件中的阈值
                if (config_manager_ && config_manager_->isLoaded()) {
                        const auto& thresholds = config_manager_->getConfig().health_change_thresholds;
                        std::string key;
                        
                        switch (waypoint_idx) {
                                case 3: key = "p4"; break;
                                case 4: key = "p5"; break;
                                case 5: key = "p6"; break;
                                case 6: case 7: case 8: case 9: case 10: key = "base"; break;
                                default: key = "default";
                        }
                        
                        auto it = thresholds.find(key);
                        if (it != thresholds.end()) {
                                return it->second;
                        }
                }
                
                // 配置未加载或找不到阈值，使用硬编码默认值
                switch (waypoint_idx) {  // 多分支
                        case 3: return HEALTH_CHANGE_THRESHOLD_P4;  // 到达P4路径点(敌方前哨站)
                        case 4: return HEALTH_CHANGE_THRESHOLD_P5;  // 到达P5路径点(补给站)
                        case 5: return HEALTH_CHANGE_THRESHOLD_P6;  // 到达P6路径点(进攻点)
                        case 6: case 7: case 8: case 9: case 10: return HEALTH_CHANGE_THRESHOLD_BASE;  // 分支
                        default: return 50;  // 返回结果
        }
    }

        int getEnemyIndex(size_t waypoint_idx) {
        // 根据操作手册的血量索引映射：
        // 索引 0-5: 蓝方（我方，若player_id=1）
        // 索引 6-11: 红方（敌方，若player_id=1）
        // Player1=蓝方，Player2=红方
        
                int enemy_index = 0;  // 将enemy_index初始化为0
        
                switch (waypoint_idx) {  // 多分支
            case 3:
                enemy_index = (player_id_ == 1) ? 9 : 3;
                                break;  // 跳出循环/分支
            case 4:
                enemy_index = (player_id_ == 1) ? 8 : 2;
                                break;  // 跳出循环/分支
            case 5:
                enemy_index = (player_id_ == 1) ? 10 : 4;
                                break;  // 跳出循环/分支
            case 6: case 7: case 8: case 9: case 10:  // P7-P11 - 攻击基地
                enemy_index = (player_id_ == 1) ? 11 : 5;
                                break;  // 跳出循环/分支
            default:
                                enemy_index = 0;
                                break;  // 跳出循环/分支
        }
        
                return enemy_index;  // 返回结果
    }
};

int main(int argc, char** argv) {
        rclcpp::init(argc, argv);  // 初始化ROS2(传入命令行参数argc和argv)

        if (argc < 2) {  // 如果条件满足,执行对应操作
                std::cerr << "Usage: " << argv[0] << " <player_id>" << std::endl;  // 输出信息
                return 1;  // 返回结果
    }

        int player_id = std::stoi(argv[1]);  // 将player_id赋值为std::stoi(argv[1])
        auto node = std::make_shared<GoNode>(player_id);  // 将node赋值为std::make_shared<GoNode>(player_id)

        rclcpp::spin(node);  // 进入ROS2事件循环(处理订阅者、发布者、定时器等)
        rclcpp::shutdown();  // 关闭ROS2(清理资源,退出程序)
        return 0;  // 正常退出程序(返回0表示成功)
}