import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from data.face_keypoint import FaceKeypointsDataset
from utils.trainer import Trainer
from utils.block import ConvBlock

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False

batch_size = 32
NUM_KEYPOINTS = 106 * 2

norm = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = FaceKeypointsDataset(
    img_dir='C:/Projects/dataset/images/train/images/merge',
    json_dir='C:/Projects/dataset/images/train/infos/merge',
    transform=norm
)
# torch.Size([3, 256, 256]) torch.Size([106 * 2])

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=False,
)

# GVV

class ImprovedConvFaceKeypoint(nn.Module):
    def __init__(self, num_classes=NUM_KEYPOINTS):
        super().__init__()
        
        # Input: 3 x 256 x 256
        self.features = nn.Sequential(
            # Block 1: -> 128 x 128
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2, 2),
            
            # Block 2: -> 64 x 64
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
            
            # Block 3: -> 32 x 32
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),
            
            # Block 4: -> 16 x 16
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
            
            # Block 5: -> 8 x 8
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(2, 2),
            
            # Block 6: -> 4 x 4
            ConvBlock(512, 512),
            nn.MaxPool2d(2, 2) 
        )
        
        # Classifier / Regressor
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 输入维度: 512 * 4 * 4 = 8192
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024), # 全连接层后也可以加 BN
            nn.ReLU(True),
            nn.Dropout(0.5),      # 防止过拟合
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, num_classes) 
        )
        
        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

model = ImprovedConvFaceKeypoint()

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

model_trainer = Trainer(
    model=model,
    num_epochs=100,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    checkpoint_path='./checkpoints/conv_face_keypoints.pth',
)

# --- 训练循环 ---
for trainer in model_trainer.train(tqdm_bar=True, print_loss=True):
    trainer.auto_update()
    trainer.auto_checkpoint()

# --- 评估循环 ---
print("\nStarting Evaluation...")
total_loss_val = 0.0
total_pixel_distance_val = 0.0
total_samples_val = 0
last_img, last_pred, last_target = None, None, None

for trainer in model_trainer.eval(description="Validating", tqdm_bar=True):
    # 维度检查
    if trainer.data.dim() == 3: trainer.data = trainer.data.unsqueeze(0)
    if trainer.target is not None and trainer.target.dim() == 1: trainer.target = trainer.target.unsqueeze(0)
    outputs = trainer.model(trainer.data)
    loss = trainer.criterion(outputs, trainer.target)
    
    batch_size_current = trainer.data.size(0)
    total_loss_val += loss.item() * batch_size_current
    
    # --- 在归一化空间计算误差 ---
    pred_points_norm = outputs.view(-1, 106, 2)
    target_points_norm = trainer.target.view(-1, 106, 2)
    distances_norm = torch.norm(pred_points_norm - target_points_norm, dim=2)
    
    total_pixel_distance_val += distances_norm.sum().item() # 这里是所有点的误差总和
    total_samples_val += batch_size_current
    # 保存最后一个 batch 的数据用于可视化
    if trainer.is_last_batch_in_epoch: # 仅保存最后一个 batch
        last_img = trainer.data[-1].detach().cpu()
        last_pred = outputs[-1].detach().cpu()
        last_target = trainer.target[-1].detach().cpu()

# --- 计算并打印最终指标 ---
avg_loss = total_loss_val / total_samples_val
# 注意：这里 total_pixel_distance_val 是所有样本 * 所有点的误差总和
# 所以要除以 (样本数 * 每张图点数) 才能得到平均每个点的误差
avg_pixel_error_norm = total_pixel_distance_val / (total_samples_val * 106)
real_avg_pixel_error = avg_pixel_error_norm * 256
print(f"\nEvaluation Completed:")
print(f"Validation Samples: {total_samples_val}")
print(f"Average Validation Loss: {avg_loss:.6f}")
print(f"Average Pixel Error: {real_avg_pixel_error:.2f} pixels")

# --- 可视化 ---
print("\nVisualizing last prediction...")
if last_img is not None:
    plt.figure(figsize=(8, 8))
    
    img_np = last_img.permute(1, 2, 0).numpy() * 0.5 + 0.5
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    pred_xy = train_data.inverse_transform_keypoints(last_pred)
    target_xy = train_data.inverse_transform_keypoints(last_target)
    
    plt.scatter(target_xy[:, 0], target_xy[:, 1], c='g', s=20, label='Ground Truth')
    plt.scatter(pred_xy[:, 0], pred_xy[:, 1], c='r', s=20, marker='x', label='Prediction')
    
    plt.legend()
    plt.title(f"Last Batch Result\nOverall Avg Error: {real_avg_pixel_error:.2f} px")
    plt.axis('off')
    plt.show()
else:
    print("No data collected for visualization.")