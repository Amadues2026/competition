/**
 * @file logger.cpp
 * @brief TensorRT 日志回调实现 —— 将 TensorRT 内部日志重定向到 stderr
 *
 * TensorRT 推理引擎在初始化和推理过程中会产生内部日志（警告、错误等），
 * 需要用户提供一个 ILogger 报告接口来接收这些日志。
 *
 * 本类实现了 nvinfer1::ILogger 接口，将 TensorRT 日志输出到标准错误流（stderr）。
 *
 * 日志级别过滤策略：
 *   - kERROR / kWARNING → 输出到 stderr（带 [Tensor ERROR/WARN] 前缀）
 *   - kINFO / kVERBOSE → 静默丢弃（避免刷屏，不影响推理结果）
 *
 * noexcept 标记：
 *   TensorRT 要求此方法不能抛出异常（否则会导致引擎崩溃），
 *   因此所有操作都在 try-catch 保护下进行（虽然当前实现本身不会抛异常）。
 *
 * 使用场景：
 *   在 YOLOv5Detector 构造函数中，创建 Logger 实例并传给 TRTLogger，
 *   用于 TensorRT 引擎反序列化（deserializeCudaEngine）过程中的错误报告。
 */

#include "logger.h"
#include <iostream>

/**
 * @brief TensorRT 日志回调（由 TensorRT 引擎内部调用）
 * @param severity 日志级别（kERROR > kWARNING > kINFO > kVERBOSE）
 * @param msg      日志消息内容（C 字符串，由 TensorRT 管理生命周期）
 *
 * 仅输出 ERROR 和 WARNING 级别的日志，INFO 和 VERBOSE 被静默忽略。
 * 输出格式：[TensorRT ERROR] 消息内容
 */
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TensorRT " << (severity == Severity::kERROR ? "ERROR" : "WARN") << "] " << msg << std::endl;
    }
}
