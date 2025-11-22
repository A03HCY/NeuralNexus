import torch
import torch.nn as nn
from utils.block import DoubleConvBlock, DownsampleBlock, UpsampleBlock




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
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=3, out_ch=1).to(device)
    
    # 创建随机输入
    x = torch.randn(1, 3, 256, 256).to(device)
    
    # 前向传播
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # 验证输出尺寸是否与输入一致
    assert x.shape[2:] == y.shape[2:], "Output spatial dimensions do not match input!"
    print("Test passed: Output dimensions match input dimensions.")
