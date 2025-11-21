import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.trainer import Trainer
from utils.block import CausalConv1d

class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples=2000, window_size=100, noise=0.1):
        """
        Args:
            num_samples: 总的时间步长数量
            window_size: 回看的时间窗口大小 (即 model 的输入 length)
            noise: 噪声强度
        """
        self.window_size = window_size
        
        # 1. 生成生成连续的时间序列数据 (Sine 波)
        # 这里的 sequence 就是我们要做预测的整个历史数据
        x_axis = np.linspace(0, 100, num_samples)
        self.data = np.sin(x_axis)
        
        # 加入噪声
        if noise > 0:
            self.data += np.random.normal(0, noise, size=num_samples)
            
        # 转换为 float32 并调整形状为 [Total_Len, 1]
        self.data = self.data.astype(np.float32).reshape(-1, 1)
    def __len__(self):
        # 样本数 = 总长度 - 窗口大小
        # 例如：总长1000，窗口100。
        # 第1个样本: idx 0~99 -> 预测 100
        # 最后一个样本: idx 899~998 -> 预测 999
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        """
        滑动窗口逻辑
        Input (Seq): data[i : i + window]
        Label (Target): data[i + window]  (预测下一个点)
        """
        # 取出窗口内的序列作为特征
        # 形状: [window_size, 1]
        sequence = self.data[idx : idx + self.window_size]
        
        # 取出窗口后的下一个点作为标签
        # 形状: [1]
        label = self.data[idx + self.window_size]
        
        return torch.from_numpy(sequence), torch.from_numpy(label)

batch_size = 32

dataset = TimeSeriesDataset(num_samples=5000, window_size=100, noise=0.1)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class TCNPredictor(nn.Module):
    def __init__(self, input_size=1, output_size=1, num_channels=[32, 32, 32, 32, 32, 32], kernel_size=3, dropout=0.2):
        """
        Args:
            input_size (int): 输入特征维度，对应 [B, 100, 1] 中的 1
            output_size (int): 预测输出维度
            num_channels (list): 每一层的隐藏层通道数。列表长度决定了层数。
                                 6层 k=3 足以覆盖 >100 的窗口。
            kernel_size (int): 卷积核大小
            dropout (float): Dropout 比率
        """
        super(TCNPredictor, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                CausalConv1d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_size, 
                    dilation=dilation_size,
                    dropout=dropout,
                    use_res=True
                )
            )
        
        self.features = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # 输入 x 的形状: [Batch, Length=100, Channel=1]
        
        # 1. 维度调整: Conv1d 接收 [Batch, Channel, Length]
        x = x.permute(0, 2, 1) 
        
        # 2. TCN 特征提取
        # 输出形状: [Batch, Hidden_Channel, Length]
        out = self.features(x)
        
        # 3. 获取最后一个时间步的信息用于预测
        # 因为是因果卷积，最后一个点包含了之前所有窗口的信息
        out = out[:, :, -1]  # 形状变为 [Batch, Hidden_Channel]
        
        # 4. 线性层输出预测值
        pred = self.linear(out) # 形状 [Batch, Output_Size]
        
        return pred


model = TCNPredictor(input_size=1, output_size=1, num_channels=[32]*6, kernel_size=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

model_trainer = Trainer(
    model=model,
    num_epochs=50,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2),
    checkpoint_path='./checkpoints/tcn_model.pt'
)

model_trainer.init_tensorboard('runs/tcn_predictor').preview_data()
model_trainer.fit(cal_predict_regression_metrics=True, vis_forecast_steps=500)