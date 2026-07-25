# D2C2 - YOLOv5n 两分类装甲板检测

## 目录结构

```
D2C2/
├── datasets/
│   └── RMCOCO2/                   # 训练数据集 (10000张, COCO格式)
│       ├── images/
│       └── annotations/
├── train/
│   ├── train_yolov5.py            # YOLOv5n 训练+导出脚本
│   └── requirements_train.txt     # Python 依赖
├── scripts/
│   └── convert_to_tensorrt.sh     # ONNX→TensorRT 转换脚本
└── README.md
```

## 训练模型

```bash
conda activate RM
pip install -r D2C2/train/requirements_train.txt
cd D2C2/train
python3 train_yolov5.py
```

训练完成后 engine 自动部署到 `models/yolov5n_d2c2_fp16.engine`。

## 模型规格

- **架构**: YOLOv5n（nano, ~2.5M 参数）
- **输入**: 224×416 RGB
- **输出**: [1, 6, 1911]（cx, cy, w, h, cls0, cls1）
- **类别**: 2 类（red / blue）
- **推理延迟**: ~2ms（TensorRT FP16）
- **mAP50**: 0.90

## 手动转 TensorRT

```bash
cd D2C2/scripts
bash convert_to_tensorrt.sh
```
