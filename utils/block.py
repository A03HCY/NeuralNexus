import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def calculate_causal_layer(step:int, kernel_size:int=3):
    if kernel_size <= 1:
        raise ValueError("kernel_size must be greater than 1")
    L = math.ceil(math.log2((step - 1) / (kernel_size - 1) + 1))
    R = 1 + (kernel_size - 1) * (2 ** L - 1)
    return int(L), R


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

class DoubleConvBlock(nn.Module):
    ''' 由两个 ConvBlock 组成的双卷积块。
    '''
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None, kernel_size: int = 3, padding: int = 1, stride: int = 1, leaky_relu: float = 0.1, act_1: bool = True, act_2: bool = True):
        ''' 初始化 DoubleConvBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            mid_ch (int, optional): 中间通道数。默认为 None（等于 out_ch）。
            kernel_size (int): 卷积核大小。默认为 3。
            padding (int): 输入两侧的零填充。默认为 1。
            stride (int): 第一个卷积层的步长。默认为 1。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
            act_1 (bool): 第一个卷积层是否使用激活函数。默认为 True。
            act_2 (bool): 第二个卷积层是否使用激活函数。默认为 True。
        '''
        super(DoubleConvBlock, self).__init__()
        mid_ch = mid_ch if mid_ch is not None else out_ch
        self.conv_1 = ConvBlock(in_ch, mid_ch, kernel_size=kernel_size, padding=padding, stride=stride, leaky_relu=leaky_relu, act=act_1)
        self.conv_2 = ConvBlock(mid_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=1, leaky_relu=leaky_relu, act=act_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class DownsampleBlock(nn.Module):
    ''' 使用带步长的 ConvBlock 或 DoubleConvBlock 进行下采样块。
    '''
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, stride: int = 2, leaky_relu: float = 0.1, act: bool = True, use_double_conv: bool = True, maxpool: bool = True, dropout_prob: float = 0.0, return_features: bool = True):
        ''' 初始化 DownsampleBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            kernel_size (int): 卷积核大小。默认为 3。
            padding (int): 输入两侧的零填充。默认为 1。
            stride (int): 卷积步长。默认为 2。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
            act (bool): 是否使用激活函数。默认为 True。
            use_double_conv (bool): 是否使用双卷积块。默认为 True。
            maxpool (bool): 是否使用 MaxPool2d(2) 进行下采样。默认为 True。
            dropout_prob (float): Dropout 概率。默认为 0.0。
            return_features (bool): 是否返回中间特征（池化前）。默认为 True。
        '''
        super(DownsampleBlock, self).__init__()
        self.return_features = return_features
        
        conv_stride = 1 if maxpool else stride
        
        if use_double_conv:
            self.block = DoubleConvBlock(in_ch, out_ch, kernel_size=kernel_size, stride=conv_stride, padding=padding, leaky_relu=leaky_relu, act_1=True, act_2=act)
        else:
            self.block = ConvBlock(in_ch, out_ch, kernel_size=kernel_size, stride=conv_stride, padding=padding, leaky_relu=leaky_relu, act=act)
            
        self.maxpool = nn.MaxPool2d(2) if maxpool else None
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        x = self.block(x)
        features = x
        if self.maxpool is not None:
            x = self.maxpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        
        if self.return_features:
            return x, features
        return x

class UpsampleBlock(nn.Module):
    ''' 使用转置卷积后接 ConvBlock 或 DoubleConvBlock 的上采样块。
    '''
    def __init__(self, in_ch: int, out_ch: int, scale_factor: int = 2, leaky_relu: float = 0.1, use_double_conv: bool = True, use_skip: bool = True):
        ''' 初始化 UpsampleBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            scale_factor (int): 空间大小的乘数。默认为 2。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
            use_double_conv (bool): 是否使用双卷积块。默认为 True。
            use_skip (bool): 是否使用跳跃连接。默认为 True。
        '''
        super(UpsampleBlock, self).__init__()
        self.use_skip = use_skip
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor)
        
        conv_in_ch = out_ch * 2 if use_skip else out_ch
        
        if use_double_conv:
            self.conv = DoubleConvBlock(conv_in_ch, out_ch, kernel_size=3, padding=1, stride=1, leaky_relu=leaky_relu)
        else:
            self.conv = ConvBlock(conv_in_ch, out_ch, kernel_size=3, padding=1, stride=1, leaky_relu=leaky_relu)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            skip (torch.Tensor, optional): 跳跃连接张量。默认为 None。

        Returns:
            torch.Tensor: 输出张量。
        '''
        x = self.up(x)
        
        if self.use_skip:
            if skip is not None:
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
            else:
                raise ValueError("UpsampleBlock expects a skip connection tensor when use_skip=True, but got None.")

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
        
        self.conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation))
        
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
    
    @staticmethod
    def auto_block(in_channels, out_channels, step, kernel_size=3, leaky_relu=0.1, use_res=True, dropout=0.2) -> nn.Sequential:
        layers, _ = calculate_causal_layer(step, kernel_size)
        model = []
        for i in range(layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else out_channels

            model.append(CausalConv1d(in_ch, out_channels, kernel_size, dilation, leaky_relu, use_res, dropout))

        return nn.Sequential(*model)
