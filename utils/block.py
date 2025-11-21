import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    ''' 带有归一化和激活函数的基本卷积块。
    '''
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, stride: int = 1, leaky_relu: float = 0.1, act: bool = True):
        ''' 初始化 ConvBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            kernel_size (int): 卷积核大小。默认为 3。
            padding (int): 输入两侧的零填充。默认为 1。
            stride (int): 卷积步长。默认为 1。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
            act (bool): 是否使用激活函数。默认为 True。
        '''
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(leaky_relu, inplace=True) if act else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class DownsampleBlock(nn.Module):
    ''' 使用带步长的 ConvBlock 进行下采样块。
    '''
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, stride: int = 2, leaky_relu: float = 0.1, act: bool = True):
        ''' 初始化 DownsampleBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            kernel_size (int): 卷积核大小。默认为 3。
            padding (int): 输入两侧的零填充。默认为 1。
            stride (int): 卷积步长。默认为 2。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
            act (bool): 是否使用激活函数。默认为 True。
        '''
        super(DownsampleBlock, self).__init__()
        self.block = ConvBlock(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, leaky_relu=leaky_relu, act=act)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        return self.block(x)

class UpsampleBlock(nn.Module):
    ''' 使用最近邻插值后接 ConvBlock 的上采样块。
    '''
    def __init__(self, in_ch: int, out_ch: int, scale_factor: int = 2, leaky_relu: float = 0.1):
        ''' 初始化 UpsampleBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            scale_factor (int): 空间大小的乘数。默认为 2。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
        '''
        super(UpsampleBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=3, padding=1, stride=1, leaky_relu=leaky_relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        x = self.up(x)
        x = self.conv(x)
        return x

class ResBasicBlock(nn.Module):
    ''' 基本残差块。
    '''
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, leaky_relu: float = 0.1):
        ''' 初始化 ResBasicBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            stride (int): 卷积步长。默认为 1。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
        '''
        super(ResBasicBlock, self).__init__()
        self.conv_1 = ConvBlock(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, leaky_relu=leaky_relu)
        self.conv_2 = ConvBlock(out_ch, out_ch, kernel_size=3, padding=1, stride=1, leaky_relu=leaky_relu, act=False)
        
        self.relu = nn.LeakyReLU(leaky_relu, inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = ConvBlock(in_ch, out_ch, kernel_size=1, padding=0, stride=stride, act=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        residual = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x + residual
        x = self.relu(x)
        return x


class ResBottleneckBlock(nn.Module):
    ''' 残差瓶颈块。
    '''
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1, leaky_relu: float = 0.1):
        ''' 初始化 ResBottleneckBlock。

        Args:
            in_ch (int): 输入通道数。
            mid_ch (int): 中间通道数。
            out_ch (int): 输出通道数。
            stride (int): 卷积步长。默认为 1。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
        '''
        super(ResBottleneckBlock, self).__init__()
        self.conv_1 = ConvBlock(in_ch, mid_ch, kernel_size=1, padding=0, leaky_relu=leaky_relu)
        self.conv_2 = ConvBlock(mid_ch, mid_ch, kernel_size=3, padding=1, stride=stride, leaky_relu=leaky_relu)

        self.conv_3 = ConvBlock(mid_ch, out_ch, kernel_size=1, padding=0, act=False)
        
        self.relu = nn.LeakyReLU(leaky_relu, inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = ConvBlock(in_ch, out_ch, kernel_size=1, padding=0, stride=stride, act=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        residual = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x + residual
        x = self.relu(x)
        return x


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, leaky_relu=0.1, use_res=True, dropout=0.2):
        super(CausalConv1d, self).__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.use_res = use_res
        
        self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation))
        
        self.relu = nn.LeakyReLU(leaky_relu, inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.downsample = None
        if use_res and in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1, padding=0)

    def forward(self, x):
        residual = x
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        if self.use_res:
            if self.downsample is not None:
                residual = self.downsample(residual)
            
            x = x + residual
            
        return x