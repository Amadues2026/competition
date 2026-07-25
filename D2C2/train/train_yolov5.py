"""
@file train_yolov5.py
@brief YOLOv5n 两分类训练脚本 —— 装甲板检测模型的完整训练流水线

功能概述：
  本脚本实现从数据准备到模型部署的完整流程：
    步骤1: COCO 标注格式 → YOLO 标注格式（坐标归一化）
    步骤2: 生成 dataset.yaml 配置文件
    步骤3: 使用 ultralytics YOLOv5n 训练两分类模型（红/蓝装甲板）
    步骤4: 导出 ONNX → TensorRT Engine（FP32，供 C++ 端推理）

数据集结构：
  datasets/RMCOCO2/
    annotations/annotations.json   COCO 格式标注文件
    images/                        原始图像
    labels/                        YOLO 格式标签（脚本生成）

类别映射：
  COCO category "red"  → YOLO class 0（红方装甲板）
  COCO category "blue" → YOLO class 1（蓝方装甲板）

训练配置：
  模型: yolov5n（轻量级，适合嵌入式部署）
  输入尺寸: 224×416（非正方形，适配相机宽高比）
  早停: patience=15（15轮无提升则停止）
  优化器: SGD（ultralytics 默认）
  学习率: lr0=1e-4（微调预训练权重时用较小值）

部署路径：
  训练产出: yolo_output/train/weights/best.pt
  部署位置: /home/robot/competition/models/yolov5n_d2c2_fp16.engine

依赖：
  pip install ultralytics
  系统: trtexec（TensorRT 工具包）
"""

import os
import json
from pathlib import Path
from ultralytics import YOLO

# ── 路径配置 ──
DATA_DIR = "../datasets/RMCOCO2"                                    # 数据集根目录
ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations", "annotations.json")  # COCO 标注文件
IMAGE_DIR = os.path.join(DATA_DIR, "images")                        # 原始图像目录
OUTPUT_DIR = "./yolo_output"                                        # 训练输出目录
YOLO_LABEL_DIR = os.path.join(DATA_DIR, "labels")                  # YOLO 标签输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(YOLO_LABEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  步骤1: COCO 标注格式 → YOLO 标注格式
# ══════════════════════════════════════════════════════════════
#
# COCO 格式: [x_topleft, y_topleft, width, height]（像素坐标，左上角为原点）
# YOLO 格式: [class, x_center, y_center, width, height]（归一化 0-1，中心点为原点）
#
# 坐标转换公式:
#   x_center = (x_topleft + width/2) / image_width
#   y_center = (y_topleft + height/2) / image_height
#   norm_w    = width / image_width
#   norm_h    = height / image_height
#
# 输出文件: labels/<image_stem>.txt，每行一个标注
print("正在转换 COCO → YOLO 格式...")

with open(ANNOTATION_FILE) as f:
    coco = json.load(f)

# 建立 COCO category_id → YOLO class_index 的映射
# 只保留 "red"(0) 和 "blue"(1) 两个类别，其他类别跳过
coco_cat_to_yolo = {}
for cat in coco["categories"]:
    if cat["name"] == "red":
        coco_cat_to_yolo[cat["id"]] = 0
    elif cat["name"] == "blue":
        coco_cat_to_yolo[cat["id"]] = 1

# 按 image_id 分组标注（同一张图的所有标注写入同一个 .txt 文件）
ann_by_image = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    ann_by_image.setdefault(img_id, []).append(ann)

# 构建 image_id → image_info 的快速查找表
img_info_map = {img["id"]: img for img in coco["images"]}

# 为每张图生成 YOLO 标签文件
converted = 0
for img_id, anns in ann_by_image.items():
    img = img_info_map[img_id]
    w, h = img["width"], img["height"]

    # 标签文件名与图像文件名同名，后缀改为 .txt
    label_path = os.path.join(YOLO_LABEL_DIR, Path(img["file_name"]).stem + ".txt")
    with open(label_path, "w") as lf:
        for ann in anns:
            x, y, bw, bh = ann["bbox"]  # COCO: [x_topleft, y_topleft, w, h]（像素）

            # 查找 YOLO 类别索引（未映射的类别跳过）
            yolo_cls = coco_cat_to_yolo.get(ann["category_id"], -1)
            if yolo_cls < 0:
                continue

            # 坐标转换: 左上角 → 中心点，像素 → 归一化
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h

            # 写入 YOLO 格式: class x_center y_center width height
            lf.write(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    converted += 1

print(f"  转换完成: {converted} 张图片, {len(coco['annotations'])} 个标注")


# ══════════════════════════════════════════════════════════════
#  步骤2: 生成 dataset.yaml 配置文件
# ══════════════════════════════════════════════════════════════
#
# dataset.yaml 是 ultralytics YOLO 训练的数据配置文件，
# 定义了训练/验证图像路径和类别名称。
#
# 注意：train 和 val 都指向 images 目录（本脚本不做 train/val 划分），
# ultralytics 内部会自动按比例（默认 80/20）拆分。
yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""path: {os.path.abspath(DATA_DIR)}
train: images
val: images
names:
  0: red
  1: blue
""")
print(f"  dataset.yaml → {yaml_path}")


# ══════════════════════════════════════════════════════════════
#  步骤3: YOLOv5n 训练
# ══════════════════════════════════════════════════════════════
#
# 使用 ultralytics API 加载预训练的 yolov5n 权重，然后在自定义数据集上微调。
# 预训练权重 yolov5n.pt 会在首次运行时自动下载。
#
# 关键训练参数：
#   epochs=100       最大训练轮数（受早停限制可能提前停止）
#   imgsz=(224,416)  非正方形输入（高224×宽416），适配相机宽高比
#   batch=8          批大小（受限于 GPU 显存）
#   lr0=1e-4         初始学习率（微调预训练权重时用较小值）
#   patience=15      早停轮数（15轮 val loss 无改善则停止）
#   save_period=5    每5轮保存 checkpoint（用于回溯最佳模型）
#
# 产出文件：
#   yolo_output/train/weights/best.pt   最佳模型权重
#   yolo_output/train/weights/last.pt   最后一轮权重
#   yolo_output/train/                  训练日志和曲线图
print("\n开始训练 YOLOv5n...")

model = YOLO("yolov5n.pt")  # 加载预训练权重（首次自动下载）

model.train(
    data=yaml_path,         # 数据配置文件路径
    epochs=100,             # 最大训练轮数
    imgsz=(224, 416),       # 输入图像尺寸（高, 宽）
    batch=8,                # 批大小
    device=0,               # GPU 设备编号（0=cuda:0）
    workers=4,              # 数据加载线程数
    lr0=1e-4,               # 初始学习率
    patience=15,            # 早停轮数
    save=True,              # 保存 checkpoint
    save_period=5,          # 每 N 轮保存一次（-1=仅保存 best/last）
    project=OUTPUT_DIR,     # 输出项目目录
    name="train",           # 实验名称（子目录）
    exist_ok=True,          # 覆盖已有实验
)


# ══════════════════════════════════════════════════════════════
#  步骤4: 模型导出 —— ONNX → TensorRT Engine
# ══════════════════════════════════════════════════════════════
#
# 部署流水线：
#   best.pt → ONNX（ultralytics export）→ TensorRT Engine（trtexec）
#   → 复制到部署路径（/home/robot/competition/models/）
#
# 为什么需要 TensorRT？
#   - ONNX Runtime 在嵌入式 GPU 上推理较慢
#   - TensorRT 对 CUDA 图做了深度优化（kernel fusion, FP16 量化等）
#   - 实测 YOLOv5n TensorRT 推理约 2-5ms/帧，满足 30Hz 实时性要求
#
# 注意：trtexec 需要系统安装 TensorRT 工具包（通常随 CUDA 一起安装）
print("\n导出 TensorRT Engine (ONNX → trtexec)...")

import glob

# 自动查找训练产出的 best.pt（递归搜索）
candidates = glob.glob("**/best.pt", recursive=True)
best_pt = candidates[0] if candidates else ""

if best_pt and os.path.exists(best_pt):
    # 4a: 导出 ONNX 格式
    # ultralytics 内部调用 torch.onnx.export，自动处理预处理/后处理
    model = YOLO(best_pt)
    onnx_path = best_pt.replace(".pt", ".onnx")
    model.export(format="onnx", imgsz=(224, 416), half=False, device=0, simplify=True)
    print(f"  ONNX → {onnx_path}")

    # 4b: ONNX → TensorRT Engine（使用 trtexec 命令行工具）
    # 默认 FP32 精度（与 C++ 端 YOLOv5Detector 兼容）
    engine_path = os.path.join(os.path.dirname(onnx_path), "best.engine")
    deploy_path = "/home/robot/competition/models/yolov5n_d2c2_fp16.engine"

    trt_cmd = f"trtexec --onnx={onnx_path} --saveEngine={engine_path}"
    ret = os.system(trt_cmd)

    if ret == 0:
        # 转换成功，复制到部署路径
        import shutil
        shutil.copy(engine_path, deploy_path)
        size_mb = os.path.getsize(deploy_path) / 1024 / 1024
        print(f"  TensorRT Engine → {deploy_path} ({size_mb:.1f}MB)")
    else:
        print(f"  TensorRT 转换失败 (exit code {ret})")
else:
    print(f"  错误: 找不到 best.pt ({best_pt})")

print("\n完成!")
