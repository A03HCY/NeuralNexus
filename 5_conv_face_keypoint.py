import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from data.face_keypoint import FaceKeypointsDataset, label_to_tensor
from utils.trainer import Trainer

batch_size = 32

norm = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = FaceKeypointsDataset(
    img_dir='C:/Projects/dataset/images/train/images/merge',
    json_dir='C:/Projects/dataset/images/train/infos/merge',
    target_transform=label_to_tensor,
    transform=norm
)
# torch.Size([3, 256, 256]) torch.Size([106 * 2])

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
)

class ConvFaceKeypoint(nn.Module):
    def __init__(self):
        super(ConvFaceKeypoint, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # 256 / 2 = 128
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 128 / 2 = 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 64 / 2 = 32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 106 * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ConvFaceKeypoint()

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

model_trainer = Trainer(
    model=model,
    num_epochs=50,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    checkpoint_path='./checkpoints/conv_face_keypoints.pth',
)

for trainer in model_trainer.train(tqdm_bar=True, print_loss=True):
    trainer.auto_update()
    trainer.auto_checkpoint()

for trainer in model_trainer.eval(train_data, tqdm_bar=True):
    trainer.test_regress()

print(f'Total Loss: {model_trainer.eval_loss:.4f}')

model_trainer.save_model('./models/conv_face_keypoints.pth')