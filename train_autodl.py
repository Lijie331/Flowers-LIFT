"""
AutoDL 训练脚本 - 花卉识别 CLIP+LIFT
使用方法：
  python train_autodl.py
"""
import os
import sys

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# CLIP缓存路径
CLIP_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'CLIP_cache')
os.environ['CLIP_CACHE_DIR'] = CLIP_CACHE_DIR

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random
import numpy as np
from collections import Counter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

print("=" * 60)
print("花卉识别训练 - CLIP + LIFT (AutoDL优化版)")
print("=" * 60)

# ==================== 配置区域 ====================
# 请根据实际情况修改这些路径
DATA_ROOT = '/root/data/ChineseFlowers120'  # 数据集路径
OUTPUT_DIR = '/root/output'                    # 输出路径
CLIP_CACHE = '/root/.cache/clip'             # CLIP缓存路径
NUM_EPOCHS = 50                              # 训练轮数
BATCH_SIZE = 32                              # 批次大小
LEARNING_RATE = 0.001                        # 学习率
# ==================== 配置结束 ====================

# 确保路径存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLIP_CACHE, exist_ok=True)

# 更新CLIP缓存路径
os.environ['CLIP_CACHE_DIR'] = CLIP_CACHE

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============= 数据增强 =============
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print(f"\n[1] 加载数据集: {DATA_ROOT}")
train_dataset = ImageFolder(DATA_ROOT, transform=TRAIN_TRANSFORM)
val_dataset = ImageFolder(DATA_ROOT, transform=VAL_TRANSFORM)

print(f"    类别数: {len(train_dataset.classes)}")
print(f"    训练样本: {len(train_dataset)}")
print(f"    验证样本: {len(val_dataset)}")

# 类别平衡采样
train_labels = train_dataset.targets
label_counts = Counter(train_labels)

print("\n[2] 创建类别平衡采样器...")
class_weights = [1.0 / label_counts[i] for i in range(len(train_dataset.classes))]
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                         num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=4, pin_memory=True)

num_classes = len(train_dataset.classes)

# ============= CLIP + LIFT 模型 =============
print("\n[3] 加载 CLIP 模型...")

from clip import clip

clip_model, preprocess = clip.load('RN50', device=device)
clip_model.float()
print(f"    CLIP backbone 已加载到 {device}")

# LIFT配置
class LIFTConfig:
    backbone = 'CLIP-RN50'
    classifier = 'CosineClassifier'
    scale = 25.0
    full_tuning = True   # 全量微调
    bias_tuning = False
    bn_tuning = True
    ssf_attn = False
    lora = False
    adapter = False

config = LIFTConfig()

from models.models import PeftModelFromCLIP

print("\n[4] 构建 LIFT 模型...")
model = PeftModelFromCLIP(config, clip_model, num_classes)
model = model.to(device)

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    总参数: {total_params:,}")
print(f"    可训练参数: {trainable_params:,}")

# 分层学习率
print("\n[5] 设置优化器...")
optimizer = torch.optim.AdamW([
    {'params': model.image_encoder.parameters(), 'lr': LEARNING_RATE * 0.1},  # backbone低学习率
    {'params': model.head.parameters(), 'lr': LEARNING_RATE}                  # head高学习率
], weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()

# AMP混合精度
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# ============= 训练函数 =============
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)
        
        if batch_idx % 20 == 0:
            print(f"    Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")
    
    return total_loss / len(loader), 100 * total_correct / total_samples

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
    
    return total_loss / len(loader), 100 * total_correct / total_samples

# ============= 开始训练 =============
print("\n[6] 开始训练...")
print("-" * 60)

best_acc = 0
best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1:2d}/{NUM_EPOCHS}]")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()
    
    lr = optimizer.param_groups[0]['lr']
    print(f"Train: {train_acc:5.2f}% | Val: {val_acc:5.2f}% | Loss: {val_loss:.4f} | LR: {lr:.6f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_acc': best_acc,
            'num_classes': num_classes,
        }, best_model_path)
        print(f"    [保存最佳模型: {best_acc:.2f}%]")

print("-" * 60)
print(f"\n训练完成! 最佳验证准确率: {best_acc:.2f}%")
print(f"模型已保存到: {best_model_path}")
print("=" * 60)
