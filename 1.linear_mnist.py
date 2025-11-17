import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
sns.set_theme(context='notebook', style='whitegrid')

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

def show_example(mnist):
    imgs = mnist.data[:12].numpy()
    labels = mnist.targets[:12]
    for i in range(12):
        img = imgs[i]
        label = labels[i].item()
        plt.subplot(3, 4, i+1)
        sns.heatmap(img, cmap='gray', cbar=False)
        plt.title(label)
        plt.axis('off')
    plt.suptitle('MNIST')
    plt.show()

# show_example(train_data)

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.linear_1 = nn.Linear(28*28, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28*28)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.linear_3(x)
        return x

model = LinearNet().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

from utils.trainer import Trainer

model_trainer = Trainer(model, num_epochs, train_loader, optimizer)
start_epoch = model_trainer.load_checkpoint('./checkpoints/linear_mnist_model.pth').display_epoch

print(f'Starting epoch: {start_epoch}')

for trainer in model_trainer.train():
    logits = model(trainer.data)
    loss = criterion(logits, trainer.target)
    trainer.update(loss)
    if trainer.is_last_batch_in_epoch:
        print(f'Epoch: {trainer.epoch}, Loss: {loss.item():.6f}')
        trainer.save_checkpoint('./checkpoints/linear_mnist_model.pth')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

torch.save(model.state_dict(), './models/linear_mnist_model.pth')
