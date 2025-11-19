import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), # BN层前不需要bias
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True) # 使用 LeakyReLU 防止梯度消失
        )
    
    def forward(self, x):
        return self.conv(x)