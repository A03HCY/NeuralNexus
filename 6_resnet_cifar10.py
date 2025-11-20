import torch
import torch.nn as nn
import torch.nn.functional as F

from data.cifar10 import train_loader, test_loader, classes
from utils.trainer import Trainer
from utils.block import ResBasicBlock


class ResClassify(nn.Module):
    def __init__(self):
        super(ResClassify, self).__init__()

        self.features = nn.Sequential(
            # 无 (stride=1), 输出: [B, 32, 32, 32]
            ResBasicBlock(3, 32, stride=1),
            ResBasicBlock(32, 32, stride=1),
            # 32x32 -> 16x16 (stride=2), 通道: 32 -> 64
            ResBasicBlock(32, 64, stride=2),
            ResBasicBlock(64, 64, stride=1),
            # 16x16 -> 8x8 (stride=2), 通道: 64 -> 128
            ResBasicBlock(64, 128, stride=2),
            ResBasicBlock(128, 128, stride=1),
            # 8x8 -> 4x4 (stride=2), 通道: 128 -> 256
            ResBasicBlock(128, 256, stride=2),
            ResBasicBlock(256, 256, stride=1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, 10)
    
    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avg(x)
        x= x.view(-1, 256)
        x = self.fc(x)
        return x

model = ResClassify()
model_trainer = Trainer(
    model=model,
    num_epochs=20,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4),
    criterion=nn.CrossEntropyLoss(),
    checkpoint_path='./checkpoints/resnet_cifar10_model.pt',
    use_amp=True,
)

model_trainer.init_tensorboard('runs/resnet_cifar10')
model_trainer.init_classes(classes, 5)
model_trainer.fit(cal_classification_metrics=True)