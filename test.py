import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from utils.trainer import Trainer

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

model = LinearNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model_trainer = Trainer(model, 1, criterion=criterion, optimizer=optimizer)
model_trainer.load_model('./models/linear_mnist_model.pth')

for trainer in model_trainer.train(train_loader, tqdm_bar=True, print_loss=True):
    trainer.auto_update()
    trainer.auto_checkpoint()

for trainer in model_trainer.eval(test_loader, tqdm_bar=True):
    trainer.test_classify()

print(trainer.eval_loss.item())

print(f'Accuracy: {100 * model_trainer.accuracy:.2f}%')