import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from data.face_keypoint import FaceKeypointsDataset
from utils.trainer import Trainer
from utils.block import ResBasicBlock, ConvBlock

batch_size = 16
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
    batch_size=1,
    shuffle=True,
)

class FaceKeyPoint(nn.Module):
    def __init__(self, num_classes=NUM_KEYPOINTS):
        super(FaceKeyPoint, self).__init__()
        # Input: 3 x 256 x 256
        self.features = nn.Sequential(
            # Block 1: -> 128 x 128
            ConvBlock(3, 32),
            ResBasicBlock(32, 32),
            nn.MaxPool2d(2, 2),
            # Block 2: -> 64 x 64
            ConvBlock(32, 64),
            ResBasicBlock(64, 64),
            nn.MaxPool2d(2, 2),
            # Block 3: -> 32 x 32
            ConvBlock(64, 128),
            ResBasicBlock(128, 128),
            nn.MaxPool2d(2, 2),
            # Block 4: -> 16 x 16
            ConvBlock(128, 256),
            ResBasicBlock(256, 256),
            nn.MaxPool2d(2, 2),
            # Block 5: -> 8 x 8
            ConvBlock(256, 512),
            ResBasicBlock(512, 512),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        return x

model = FaceKeyPoint()
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

model_trainer = Trainer(
    model=model,
    num_epochs=50,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    checkpoint_path='./checkpoints/face_keypoint_model.pt',
)

model_trainer.init_tensorboard('runs/face_keypoint')
model_trainer.fit()

last_img, last_pred, last_target = None, None, None

last_loss = 1

for trainer in model_trainer.eval(tqdm_bar=True):
    if not trainer.is_last_batch_in_epoch: continue
    if trainer.data.dim() == 3: trainer.data = trainer.data.unsqueeze(0)
    if trainer.target is not None and trainer.target.dim() == 1: trainer.target = trainer.target.unsqueeze(0)
    output: torch.Tensor = trainer.model.forward(trainer.data)
    loss = trainer.criterion(output, trainer.target)
    if last_loss > loss.item():
        last_loss = loss.item()
        last_img = trainer.data[-1].detach().to('cpu')
        last_pred = output[-1].detach().to('cpu')
        last_target = trainer.target[-1].detach().to('cpu')

print("Visualizing last prediction...")
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
    plt.title(f"Last Batch Result")
    plt.axis('off')
    plt.show()
else:
    print("No data collected for visualization.")