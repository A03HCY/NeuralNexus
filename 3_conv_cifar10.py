import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from utils.trainer import Trainer

torch.manual_seed(42)
batch_size = 128

norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(
    root='../data/cifar10',
    train=True,
    download=True,
    transform=norm
)
test_data = torchvision.datasets.CIFAR10(
    root='../data/cifar10',
    train=False,
    download=True,
    transform=norm
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
)
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=False,
)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model_trainer = Trainer(model, 15, train_loader=train_loader, optimizer=optimizer)
start_epoch = model_trainer.load_checkpoint('./checkpoints/conv_cifar10_model.pth').display_epoch

for trainer in model_trainer.train():
    logist = model.forward(trainer.data)
    loss = criterion(logist, trainer.target)
    trainer.update(loss)
    if trainer.is_last_batch_in_epoch:
        print(f'Epoch: {trainer.display_epoch}, Loss: {loss.item():.6f}')
        trainer.save_checkpoint('./checkpoints/conv_cifar10_model.pth')

correct = 0
total = 0
for trainer in model_trainer.eval(test_loader, tqdm_bar=True):
    logist = model.forward(trainer.data)
    _, predicted = torch.max(logist.data, 1)
    total += trainer.target.size(0)
    correct += (predicted == trainer.target).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

model_trainer.save_model('./models/conv_cifar10_model.pth')