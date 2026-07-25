/**
 * @file logger.h
 * @brief TensorRT 日志回调接口声明 —— 实现 nvinfer1::ILogger 用于接收引擎内部日志
 *
 * TensorRT 推理引擎在初始化（反序列化引擎）和推理过程中会产生内部日志，
 * 需要用户提供 ILogger 接口来接收这些日志。
 *
 * 本类继承 nvinfer1::ILogger，重写 log() 方法将日志输出到 stderr。
 * 仅输出 ERROR 和 WARNING 级别，INFO/VERBOSE 静默忽略。
 *
 * 使用场景：
 *   YOLOv5Detector 构造函数中创建 Logger 实例，传给 TRTLogger 用于引擎初始化。
 *
 * 注意：log() 方法标记为 noexcept（TensorRT 要求），不能抛出异常。
 */

#ifndef COMPETITION_LOGGER_H
#define COMPETITION_LOGGER_H

#include <NvInfer.h>

class Logger : public nvinfer1::ILogger {
    /**
     * @brief TensorRT 日志回调（由引擎内部调用）
     * @param severity 日志级别（kERROR > kWARNING > kINFO > kVERBOSE）
     * @param msg      日志消息（C 字符串，由 TensorRT 管理生命周期）
     */
    void log(Severity severity, const char* msg) noexcept override;
};

#endif
