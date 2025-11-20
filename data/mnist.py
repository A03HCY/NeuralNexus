import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

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