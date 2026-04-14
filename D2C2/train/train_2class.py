#!/usr/bin/env python3
"""
CenterNet两分类模型训练脚本
基于ResNet18 backbone,输出2个类别(red/blue)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import tensorrt as trt
import onnx
import onnxruntime as ort

# ==================== 模型定义 ====================
class BasicBlock(nn.Module):
    """BasicBlock for ResNet18"""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet18CenterNet(nn.Module):
    """CenterNet with ResNet18 backbone for 2-class detection"""
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet18CenterNet, self).__init__()
        self.num_classes = num_classes
        
        # BasicBlock for ResNet18
        self.inplanes = 64
        
        # Stem (下采样4倍: conv1 stride=2 + maxpool stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet18 layers (保持输出为输入的1/4)
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 2, stride=1)  # /4 -> /4
        self.layer2 = self._make_layer(BasicBlock, 64, 128, 2, stride=1) # /4 -> /4
        
        # Heads for CenterNet (输出名称与原模型一致: fmap, reg, wh)
        # 输入: 128 channels, output size: /4 of input
        self.fmap_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)  # 2 classes: red, blue
        )
        
        self.reg_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1)  # x, y offset
        )
        
        self.wh_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1)  # width, height
        )
        
        # 初始化heads
        self._init_head(self.fmap_head)
        self._init_head(self.reg_head)
        self._init_head(self.wh_head)
        
        # 初始化权重
        self._init_weights()
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def _init_head(self, head):
        """Initialize head layers"""
        for m in head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone (下采样4倍)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Heads (输出名称与原模型一致: fmap, reg, wh)
        fmap = self.fmap_head(x)
        reg = self.reg_head(x)
        wh = self.wh_head(x)
        
        return fmap, reg, wh

# ==================== 自适应高斯半径 ====================
def gaussian_radius(det_size, min_overlap=0.7):
    """
    计算自适应高斯半径
    根据目标大小动态计算最佳半径，使高斯核能够覆盖目标中心
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)

def draw_gaussian(fmap, center, radius):
    """
    在热力图上绘制高斯分布
    """
    diameter = 2 * radius + 1
    gaussian = np.exp(-(np.arange(diameter) - radius) ** 2 / (2.0 * (radius / 3.0) ** 2))

    x, y = int(center[0]), int(center[1])
    height, width = fmap.shape[:2]

    for i in range(diameter):
        for j in range(diameter):
            nx, ny = x - radius + i, y - radius + j
            if 0 <= nx < width and 0 <= ny < height:
                fmap[ny, nx] = max(fmap[ny, nx], gaussian[i] * gaussian[j])

# ==================== 数据集 ====================
class COCODataset(Dataset):
    def __init__(self, annotation_file, image_dir, input_size=(224, 416), output_size=(56, 104), augment=True):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.input_w, self.input_h = input_size
        self.output_w, self.output_h = output_size
        self.augment = augment  # 是否使用数据增强
        self.images = self.coco.getImgIds()

        # 获取所有类别
        self.cat_ids = self.coco.getCatIds()
        print(f"数据集类别数: {len(self.cat_ids)}")
        print(f"数据集图片数: {len(self.images)}")
        print(f"数据增强: {'开启' if augment else '关闭'}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # 读取图像
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 数据增强
        if self.augment and np.random.random() > 0.5:
            # 随机水平翻转
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)
                # 同时翻转bbox
                for ann in anns:
                    bbox = ann['bbox']
                    bbox[0] = img_info['width'] - bbox[0] - bbox[2]

            # 随机颜色抖动
            if np.random.random() > 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                hsv = hsv.astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-10, 10)) % 180  # 色调
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)  # 饱和度
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)  # 亮度
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 调整大小 (与原模型一致: 224x416)
        img = cv2.resize(img, (self.input_w, self.input_h))

        # 预处理 (与推理代码一致)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = torch.from_numpy(img).float().permute(2, 0, 1)

        # 生成ground truth (下采样4倍: 224/4=56, 416/4=104)
        fmap = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
        reg = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
        wh = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)

        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            cat_id = ann['category_id']  # 0=red, 1=blue

            # 转换到output坐标系
            x, y, w, h = bbox
            x = x / img_info['width'] * self.input_w
            y = y / img_info['height'] * self.input_h
            w = w / img_info['width'] * self.input_w
            h = h / img_info['height'] * self.input_h

            # 转换到output size (下采样4倍)
            cx = (x + w / 2) / 4
            cy = (y + h / 2) / 4
            ow = w / 4  # 存储缩放后的宽高，推理时会乘以4还原
            oh = h / 4

            # 计算整数坐标和小数偏移
            cx_int = int(cx)
            cy_int = int(cy)
            reg_x = cx - cx_int
            reg_y = cy - cy_int

            # 边界检查
            cx_int = max(0, min(cx_int, self.output_w - 1))
            cy_int = max(0, min(cy_int, self.output_h - 1))

            # 使用自适应高斯半径
            radius = gaussian_radius((ow, oh), min_overlap=0.7)
            radius = max(2, int(radius))  # 至少为2
            radius = min(radius, 10)  # 最多为10

            # 在fmap上画高斯
            draw_gaussian(fmap[cat_id], (cx_int, cy_int), radius)

            reg[0, cy_int, cx_int] = reg_x
            reg[1, cy_int, cx_int] = reg_y
            wh[0, cy_int, cx_int] = ow
            wh[1, cy_int, cx_int] = oh

        fmap = torch.from_numpy(fmap).float()
        reg = torch.from_numpy(reg).float()
        wh = torch.from_numpy(wh).float()

        return img, fmap, reg, wh

# ==================== 损失函数 ====================
class CenterNetLoss(nn.Module):
    def __init__(self):
        super(CenterNetLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_fmap, pred_reg, pred_wh,
                 target_fmap, target_reg, target_wh):
        # Fmap loss (Focal Loss) - 热力图损失权重为1.0
        fmap_loss = self.focal_loss(pred_fmap, target_fmap)

        # Reg loss (L1,只在有目标的点上计算) - 位置偏移损失权重降低为0.1
        mask = (target_fmap.sum(dim=1, keepdim=True) > 0).float()
        reg_loss = 0.1 * self.l1_loss(pred_reg * mask, target_reg * mask)

        # WH loss (L1,只在有目标的点上计算) - 宽高损失权重降低为0.1
        wh_loss = 0.1 * self.l1_loss(pred_wh * mask, target_wh * mask)

        # 总损失 = 热力图损失 + 0.1*位置偏移损失 + 0.1*宽高损失
        # 降低回归损失权重可以提升检测稳定性，减少"一掉一掉"现象
        total_loss = fmap_loss + reg_loss + wh_loss

        return total_loss, fmap_loss, reg_loss, wh_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
        pt = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce
        
        return focal_loss.mean()

# ==================== 训练函数 ====================
def train():
# 训练配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    num_epochs = 100  # 增加训练轮数到100，提升模型稳定性
    learning_rate = 1e-4
    input_size = (224, 416)
    output_size = (56, 104)

    # 数据路径
    data_dir = '../datasets'
    annotation_file = os.path.join(data_dir, 'RMCOCO2', 'annotations', 'annotations.json')
    image_dir = os.path.join(data_dir, 'RMCOCO2', 'images')

    # 创建模型
    model = ResNet18CenterNet(num_classes=2, pretrained=True)
    model = model.to(device)

    # 数据集和数据加载器（启用数据增强）
    dataset = COCODataset(annotation_file, image_dir, input_size, output_size, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 损失函数和优化器
    criterion = CenterNetLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)  # 每40个epoch学习率降低10倍
    
    # 检查是否有checkpoint并恢复
    start_epoch = 0
    checkpoint_dir = './models'
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('centernet_2class_epoch') and f.endswith('.pth')],
                             key=lambda x: int(x.split('epoch')[1].split('.pth')[0]))

    if checkpoint_files:
        # 加载最新的checkpoint
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        print(f"从checkpoint恢复训练: {latest_checkpoint}")
        print(f"恢复到Epoch {start_epoch}/{num_epochs}")
        print(f"上次的损失值: {checkpoint['loss']:.4f}")
    
    # 训练
    print(f"开始训练...")
    print(f"设备: {device}")
    print(f"输入尺寸: {input_size}")
    print(f"输出尺寸: {output_size}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {num_epochs}")
    
    model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_fmap_loss = 0
        epoch_reg_loss = 0
        epoch_wh_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, fmaps, regs, whs) in enumerate(pbar):
            images = images.to(device)
            fmaps = fmaps.to(device)
            regs = regs.to(device)
            whs = whs.to(device)
            
            # 前向传播
            pred_fmap, pred_reg, pred_wh = model(images)
            
            # 计算损失
            loss, fmap_loss, reg_loss, wh_loss = criterion(
                pred_fmap, pred_reg, pred_wh,
                fmaps, regs, whs
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            epoch_fmap_loss += fmap_loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_wh_loss += wh_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'fmap': f'{fmap_loss.item():.4f}',
                'reg': f'{reg_loss.item():.4f}',
                'wh': f'{wh_loss.item():.4f}'
            })
        
        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

        # 每20个epoch保存一次checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f'./models/centernet_2class_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")

    # 保存最终模型
    final_model_path = './models/centernet_2class_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"训练完成! 最终模型保存至: {final_model_path}")
    
    return model

# ==================== 导出ONNX ====================
def export_to_onnx(model, output_path='./models/centernet_2class.onnx'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    
    # 创建示例输入 (与原模型一致: 224x416)
    dummy_input = torch.randn(1, 3, 224, 416).to(device)
    
    # 导出ONNX (输出名称与原模型一致: fmap, reg, wh)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['fmap', 'reg', 'wh'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'fmap': {0: 'batch_size'},
            'reg': {0: 'batch_size'},
            'wh': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX模型导出至: {output_path}")
    
    # 验证ONNX模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过")
    
    # 测试推理
    ort_session = ort.InferenceSession(output_path)
    outputs = ort_session.run(None, {'input': dummy_input.cpu().numpy()})
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"fmap输出形状: {outputs[0].shape}")
    print(f"reg输出形状: {outputs[1].shape}")
    print(f"wh输出形状: {outputs[2].shape}")

# ==================== 主函数 ====================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'export'])
    parser.add_argument('--model_path', type=str,
                       default='./models/centernet_2class_final.pth')
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 训练模型
        model = train()
        
        # 导出ONNX
        export_to_onnx(model)
    else:
        # 仅导出ONNX
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet18CenterNet(num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(args.model_path))
        export_to_onnx(model)