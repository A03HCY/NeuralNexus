# trainer.py

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Any, Generator, Iterator, Optional, Dict

class Trainer:
    """
    一个全功能的训练迭代器，封装了设备管理、状态跟踪、参数更新和检查点逻辑。

    通过迭代 Trainer 实例，可以简化训练循环，并通过其属性访问当前状态
    (epoch, batch_idx, data, target等)，同时提供 update() 和 checkpointing 方法。
    """
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int,
        data_loader: DataLoader,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        初始化 Trainer。

        Args:
            model (nn.Module): 需要训练的模型。
            num_epochs (int): 总的训练轮数。
            data_loader (DataLoader): 训练数据加载器。
            optimizer (Optional[Optimizer]): 优化器实例。如果提供，可使用 trainer.update()。
            device (Optional[torch.device]): 计算设备。默认为自动检测CUDA。
        """
        self.model = model
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.optimizer = optimizer
        
        # --- 设备管理 (Device Management) ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Trainer initialized. Using device: {self.device}")
        self.model.to(self.device)

        # --- 状态变量 (State Variables) ---
        self.epoch: int = 0
        self.batch_idx: int = 0
        self.data: Optional[Any] = None
        self.target: Optional[Any] = None
        self.is_last_batch_in_epoch: bool = False
        self.start_epoch: int = 0  # 用于恢复训练的起始 epoch
        
        self._iterator: Iterator['Trainer'] = self._create_iterator()

    def _create_iterator(self) -> Generator['Trainer', None, None]:
        """内部生成器，实现核心的循环逻辑。"""
        self.model.train()
        num_batches = len(self.data_loader)
        
        # 循环从 start_epoch 开始，确保可以恢复训练
        for epoch_num in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch_num  # 更新当前 epoch
            
            for batch_idx, batch_data in enumerate(self.data_loader):
                # 更新所有批次相关的状态
                self.batch_idx = batch_idx
                self.data = batch_data[0].to(self.device)
                self.target = batch_data[1].to(self.device)
                self.is_last_batch_in_epoch = (batch_idx == num_batches - 1)
                
                yield self

    def __iter__(self) -> Iterator['Trainer']:
        """返回迭代器对象，使其可以被用于 for 循环。"""
        return self._iterator

    def update(self, loss: torch.Tensor) -> None:
        """
        执行反向传播和优化器步骤。
        封装了 optimizer.zero_grad(), loss.backward(), optimizer.step()。
        """
        if self.optimizer is None:
            raise RuntimeError(
                "Cannot call update() because no optimizer was provided to the Trainer."
            )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_checkpoint(self, path: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
        """
        保存训练检查点，如果目录不存在则会自动创建。
        保存的 'epoch' 是当前已完成的 epoch 编号。
        """
        # 确保保存检查点的目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,  # 保存当前已完成的 epoch
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        }
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """
        加载训练检查点以恢复训练。
        """
        if not os.path.exists(path):
            return self.start_epoch

        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 设置下一个 epoch 作为起始点
            self.start_epoch = checkpoint['epoch'] + 1
        except:
            self.start_epoch = 0
        
        return self.start_epoch