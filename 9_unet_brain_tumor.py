import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from utils.block import DoubleConvBlock, DownsampleBlock, UpsampleBlock
from data.brain_tumor import train_dataset, valid_dataset, test_dataset, test_path
from utils.trainer import Trainer

batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class UNet(nn.Module):
    ''' 使用 utils.block 构建的简单 U-Net 模型。
    '''
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        ''' 初始化 SimpleUNet。

        Args:
            in_ch (int): 输入通道数。默认为 3。
            out_ch (int): 输出通道数。默认为 1。
        '''
        super(UNet, self).__init__()

        # Encoder (下采样路径)
        # DownsampleBlock 内部执行: Conv -> Features(Skip) -> MaxPool -> Output
        self.down1 = DownsampleBlock(in_ch, 64)
        self.down2 = DownsampleBlock(64, 128)
        self.down3 = DownsampleBlock(128, 256)
        self.down4 = DownsampleBlock(256, 512)

        # Bottleneck (瓶颈层)
        self.bottleneck = DoubleConvBlock(512, 1024)

        # Decoder (上采样路径)
        # UpsampleBlock 内部执行: UpConv -> Concat(Skip) -> Conv
        self.up1 = UpsampleBlock(1024, 512)
        self.up2 = UpsampleBlock(512, 256)
        self.up3 = UpsampleBlock(256, 128)
        self.up4 = UpsampleBlock(128, 64)

        # Output layer
        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量 (B, C, H, W)。

        Returns:
            torch.Tensor: 输出张量 (B, Out_C, H, W)。
        '''
        # Encoder
        # x_pooled 是下采样后的输出，传给下一层
        # x_skip 是下采样前的特征，用于跳跃连接
        x1, f1 = self.down1(x)  # f1: 64 ch, x1: 64 ch (pooled)
        x2, f2 = self.down2(x1) # f2: 128 ch, x2: 128 ch (pooled)
        x3, f3 = self.down3(x2) # f3: 256 ch, x3: 256 ch (pooled)
        x4, f4 = self.down4(x3) # f4: 512 ch, x4: 512 ch (pooled)

        # Bottleneck
        x = self.bottleneck(x4) # 1024 ch

        # Decoder
        x = self.up1(x, f4) # 1024 -> 512 ch
        x = self.up2(x, f3) # 512 -> 256 ch
        x = self.up3(x, f2) # 256 -> 128 ch
        x = self.up4(x, f1) # 128 -> 64 ch

        # Output
        out = self.outc(x)
        logits = self.sigmoid(out)
        return logits

model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

model_trainer = Trainer(
    model=model,
    num_epochs=10,
    train_loader=train_loader,
    test_loader=valid_loader,
    optimizer=optimizer,
    criterion=criterion,
    checkpoint_path='checkpoints/brain_tumor.pth'
)

model_trainer.init_tensorboard(log_dir='runs/brain_tumor').preview_data()
model_trainer.fit()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')

    display_image = image.resize((256, 256), Image.Resampling.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image)
    return image_tensor.unsqueeze(0), display_image

def predict_mask(model, image_tensor, device='cuda', threshold=0.5):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)
        prediction = (prediction > threshold).float()
    return prediction

def visualize_result(original_image, predicted_mask):
    plt.subplot(131)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(np.array(original_image))
    plt.imshow(predicted_mask.squeeze(), cmap='Reds', alpha=0.4)
    plt.title('Overlay')
    plt.axis('off')

samp = test_path + '/' + '1203_jpg.rf.be8a48f34842f2c23a84b8b367618dce.jpg'
image_tensor, original_image = preprocess_image(samp)
predicted_mask = predict_mask(model, image_tensor, device='cuda')
predicted_mask = predicted_mask.cpu().numpy()
visualize_result(original_image, predicted_mask)