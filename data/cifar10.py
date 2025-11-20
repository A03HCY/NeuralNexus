import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')