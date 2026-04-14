#!/bin/bash

# TensorRT引擎转换脚本
# 将ONNX模型转换为TensorRT FP16引擎

set -e  # 遇到错误立即退出

# 配置
ONNX_PATH="../train/models/centernet_2class.onnx"
ENGINE_PATH="../inference/centernet_r18d2c2_fp16.engine"

echo "=========================================="
echo "TensorRT引擎转换脚本"
echo "=========================================="

# 检查ONNX文件是否存在
if [ ! -f "$ONNX_PATH" ]; then
    echo "错误: ONNX文件不存在: $ONNX_PATH"
    echo "请先完成训练以生成ONNX模型"
    exit 1
fi

echo "ONNX文件: $ONNX_PATH"
echo "输出引擎: $ENGINE_PATH"
echo ""

# 删除旧的engine文件（如果存在）
if [ -f "$ENGINE_PATH" ]; then
    echo "删除旧的engine文件..."
    rm -f "$ENGINE_PATH"
fi

# 执行转换
echo "开始转换ONNX到TensorRT FP16引擎..."
echo "这可能需要几分钟..."
echo ""

trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --fp16

# 检查转换是否成功
if [ -f "$ENGINE_PATH" ]; then
    ENGINE_SIZE=$(du -h "$ENGINE_PATH" | cut -f1)
    echo ""
    echo "=========================================="
    echo "✅ 转换成功！"
    echo "=========================================="
    echo "Engine文件: $ENGINE_PATH"
    echo "文件大小: $ENGINE_SIZE"
    echo ""
    echo "✅ TensorRT 引擎已生成！"
else
    echo ""
    echo "=========================================="
    echo "❌ 转换失败！"
    echo "=========================================="
    echo "请检查trtexec输出日志"
    exit 1
fi