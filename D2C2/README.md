# D2C2 - CenterNet 双分类装甲板检测模型

## 目录结构

```
D2C2/
├── datasets/                       # 数据集
│   ├── RMCOCO2/                   # 训练数据集
│   │   ├── images/
│   │   └── annotations/
│   ├── MSCOCO/                    # COCO 数据集
│   └── RMRobot/                   # 机器人数据集
├── train/                          # 训练库
│   ├── train_2class.py            # 训练脚本
│   ├── requirements_train.txt     # Python 依赖
│   └── models/                    # 训练输出目录
│       ├── centernet_2class_epoch20.pth
│       ├── centernet_2class_epoch40.pth
│       ├── centernet_2class_epoch60.pth
│       ├── centernet_2class_epoch80.pth
│       ├── centernet_2class_epoch100.pth
│       ├── centernet_2class_final.pth
│       └── centernet_2class.onnx  # ONNX 模型
├── inference/                      # 推理库
│   └── centernet_r18d2c2_fp16.engine  # TensorRT 引擎（最终模型）
├── scripts/                        # 脚本工具
│   └── convert_to_tensorrt.sh     # TensorRT 转换脚本
└── README.md                       # 本文件
```

## 如何得到最终模型

### 1. 安装依赖

```bash
cd D2C2/train
pip install -r requirements_train.txt
```

### 2. 训练模型

训练 100 个 epoch，模型会自动保存到 `train/models/` 目录：

```bash
cd D2C2/train
python3 train_2class.py
```

**输出**:
- Checkpoint 文件: `train/models/centernet_2class_epoch{N}.pth`
- 最终模型: `train/models/centernet_2class_final.pth`

**训练特性**:
- 自适应高斯半径
- 数据增强（水平翻转、颜色抖动）
- 自动 checkpoint 恢复
- 损失权重: `fmap_loss + 0.1*reg_loss + 0.1*wh_loss`

### 3. 导出 ONNX 模型

训练完成后会自动导出 ONNX 模型到 `train/models/centernet_2class.onnx`。

或者手动导出：

```bash
cd D2C2/train
python3 train_2class.py --export
```

### 4. 转换为 TensorRT 引擎

```bash
cd D2C2/scripts
bash convert_to_tensorrt.sh
```

**输出**: `inference/centernet_r18d2c2_fp16.engine`

**性能**:
- GPU 延迟: 0.38ms
- 主机延迟: 0.52ms
- 文件大小: 2.6 MB

## 数据集准备

数据集已包含在 `datasets/RMCOCO2/` 目录：

```
datasets/RMCOCO2/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── annotations/
    └── annotations.json
```

使用 COCO 格式标注。

## 模型规格

- **骨干网络**: ResNet18
- **输入尺寸**: 从模型文件自动读取（典型值：224x416）
- **输出尺寸**: 从模型文件自动读取（典型值：56x104，4x 下采样）
- **类别**: 2 类 (红色装甲板、蓝色装甲板)
- **推理速度**: 0.38ms (TensorRT FP16)

**注意**: 实际输入/输出尺寸由训练时的配置决定，运行时从TensorRT引擎文件中读取。