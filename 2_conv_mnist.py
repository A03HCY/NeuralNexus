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
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = torchvision.datasets.MNIST(
    root='../data/mnist',
    train=True,
    download=True,
    transform=norm
)
test_data = torchvision.datasets.MNIST(
    root='../data/mnist',
    train=False,
    download=True,
    transform=norm
)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ConvNet()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model_trainer = Trainer(model, 10, train_loader, optimizer=optimizer)
start_epoch = model_trainer.load_checkpoint('./checkpoints/conv_mnist_model.pth').display_epoch

for trainer in model_trainer.train():
    logist = model.forward(trainer.data)
    loss = criterion(logist, trainer.target)
    trainer.update(loss)
    if trainer.is_last_batch_in_epoch:
        print(f'Epoch: {trainer.epoch}, Loss: {loss.item():.6f}')
        trainer.save_checkpoint('./checkpoints/conv_mnist_model.pth')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(model_trainer.device), target.to(model_trainer.device)
        logits = model(data)
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

torch.save(model.state_dict(), './models/conv_mnist_model.pth')