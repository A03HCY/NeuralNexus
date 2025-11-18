import os
import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Any, Generator, Iterator, Optional, Dict
from tqdm import tqdm

from torch.optim.lr_scheduler import _LRScheduler

class Trainer:
    """
    一个全功能的训练/评估迭代器，封装了设备管理、状态跟踪、参数更新和检查点逻辑。
    通过迭代 Trainer 实例，可以简化训练和评估循环，并通过其属性访问当前状态
    (epoch, batch_idx, data, target等)，同时提供 update() 和 checkpointing 方法。
    """
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int,
        train_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        初始化 Trainer。

        Args:
            model (nn.Module): 需要训练和评估的模型。
            num_epochs (int): 总的训练轮数。
            train_loader (Optional[DataLoader]): 默认的训练数据加载器。
            test_loader (Optional[DataLoader]): 默认的测试/评估数据加载器。
            optimizer (Optional[Optimizer]): 优化器实例。如果提供，可使用 trainer.update()。
            criterion (Optional[nn.Module]): 损失函数。如果与 optimizer 一同提供，可使用 trainer.auto_update()。
            scheduler (Optional[_LRScheduler]): 学习率调度器。如果提供，可通过 trainer.auto_step_scheduler() 更新。
            checkpoint_path (Optional[str]): 默认的检查点文件路径。如果提供，初始化时会自动加载检查点。
            device (Optional[torch.device]): 计算设备。默认为自动检测CUDA。
        """
        self.model = model
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path

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
        self.start_epoch: int = 0

        if self.checkpoint_path is not None:
            self.load_checkpoint()

    @property
    def display_epoch(self) -> int:
        """返回从 1 开始计数的、用于向用户展示的当前 epoch 编号。"""
        epoch = max(self.epoch, self.start_epoch)
        return epoch + 1

    def _create_train_iterator(self, data_loader: DataLoader) -> Generator['Trainer', None, None]:
        """内部生成器，为 train() 方法实现核心的循环逻辑。"""
        self.model.train()
        num_batches = len(data_loader)
        
        for epoch_num in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch_num
            
            for batch_idx, batch_data in enumerate(data_loader):
                self.batch_idx = batch_idx
                # 将除了最后一个元素之外的所有内容都视为 data
                # 将最后一个元素视为 target
                if isinstance(batch_data, (list, tuple)) and len(batch_data) > 1:
                    # 将数据和目标移动到设备
                    self.data = tuple(d.to(self.device) for d in batch_data[:-1])
                    self.target = batch_data[-1].to(self.device)
                    # 如果只有一个数据元素，则不使用元组
                    if len(self.data) == 1:
                        self.data = self.data[0]
                else: # 兼容原始的 (data, target) 格式
                    self.data = batch_data[0].to(self.device)
                    self.target = batch_data[1].to(self.device)
                
                self.is_last_batch_in_epoch = (batch_idx == num_batches - 1)
                
                yield self

    def train(self, train_loader: Optional[DataLoader] = None) -> Iterator['Trainer']:
        """
        创建一个用于训练的迭代器。

        可以在运行时传入一个 train_loader 临时使用，否则会使用初始化时提供的
        默认 train_loader。如果两者都未提供，则会报错。
        """
        loader_to_use = train_loader if train_loader is not None else self.train_loader
        if loader_to_use is None:
            raise ValueError(
                "No train_loader available. Please provide one to the train() method "
                "or during Trainer initialization."
            )
        return self._create_train_iterator(loader_to_use)
    
    def _create_eval_iterator(self, data_loader: DataLoader, description: str, tqdm_bar: bool) -> Generator['Trainer', None, None]:
        """内部生成器，为 eval() 方法实现核心的循环逻辑。"""
        self.model.eval()
        try:
            iterable = tqdm(data_loader, desc=description, leave=False) if tqdm_bar else data_loader
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(iterable):
                    self.batch_idx = batch_idx
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) > 1:
                        self.data = tuple(d.to(self.device) for d in batch_data[:-1])
                        self.target = batch_data[-1].to(self.device)
                        if len(self.data) == 1:
                            self.data = self.data[0]
                    else:
                        self.data = batch_data[0].to(self.device)
                        self.target = batch_data[1].to(self.device)
                    yield self
        finally:
            self.model.train()

    def eval(self, 
             test_loader: Optional[DataLoader] = None, 
             description: str = "Evaluating", 
             tqdm_bar: bool = False
            ) -> Generator['Trainer', None, None]:
        """
        创建一个用于评估的迭代器。

        可以在运行时传入一个 test_loader 临时使用，否则会使用初始化时提供的
        默认 test_loader。如果两者都未提供，则会报错。
        """
        loader_to_use = test_loader if test_loader is not None else self.test_loader
        if loader_to_use is None:
            raise ValueError(
                "No test_loader available. Please provide one to the eval() method "
                "or during Trainer initialization."
            )
        return self._create_eval_iterator(loader_to_use, description, tqdm_bar)

    def step_scheduler(self, metric: Optional[float] = None) -> None:
        """
        更新学习率调度器。
        如果调度器是 ReduceLROnPlateau，则需要提供 metric 参数。
        如果调度器是其他类型（如 StepLR、CosineAnnealingLR 等），则不需要提供 metric 参数。

        Args:
            metric (Optional[float]): 用于 ReduceLROnPlateau 的指标值。

        Raises:
            ValueError: 如果调度器是 ReduceLROnPlateau 但没有提供 metric。
            RuntimeError: 如果没有在 Trainer 中配置调度器。
        """
        if self.scheduler is None:
            # 如果没有调度器，静默返回或打印警告
            # print("Warning: step_scheduler() called but no scheduler is configured.")
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError(
                    "A metric (e.g., validation loss) must be provided for ReduceLROnPlateau scheduler."
                )
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def auto_step_scheduler(self, loss: Optional[torch.Tensor] = None, step_plateau_with_train_loss: bool = False) -> None:
        """
        在每个 epoch 结束时，有条件地自动更新学习率调度器。
        此方法会检查当前是否为 epoch 的最后一个批次。如果是，则会根据调度器类型执行更新：
        - 对于 `StepLR` 等，会自动调用 `step()`。
        - 对于 `ReduceLROnPlateau`，默认不执行。
        - 若要强制使用当前训练损失更新 `ReduceLROnPlateau`，需将 `step_plateau_with_train_loss` 设为 True，
          此时必须提供 `loss` 参数。
        Args:
            loss (Optional[torch.Tensor]): 当前批次的损失张量。对于 ReduceLROnPlateau 是必需的。
            step_plateau_with_train_loss (bool): 是否强制更新 `ReduceLROnPlateau`。
        """
        if self.is_last_batch_in_epoch and self.scheduler is not None:
            is_plateau_scheduler = isinstance(self.scheduler, ReduceLROnPlateau)
            # 1. 如果不是 Plateau 调度器，则总是更新。
            # 2. 如果是 Plateau 调度器，则仅在用户显式要求时才更新。
            if not is_plateau_scheduler or step_plateau_with_train_loss:
                self.step_scheduler(loss.item())

    def update(self, loss: torch.Tensor, step_plateau_with_train_loss: bool = False) -> None:
        """
        执行反向传播和优化器步骤，并在 epoch 结束时自动更新调度器。
        Args:
            loss (torch.Tensor): 要反向传播的损失张量。
            step_plateau_with_train_loss (bool): 是否强制使用此 `loss` 更新 `ReduceLROnPlateau`。
        """
        if self.optimizer is None:
            raise RuntimeError(
                "Cannot call update() because no optimizer was provided to the Trainer."
            )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 调用专用的自动调度器步进方法
        self.auto_step_scheduler(loss, step_plateau_with_train_loss=step_plateau_with_train_loss)
    
    def auto_update(self, step_plateau_with_train_loss: bool = False) -> torch.Tensor:
        """
        执行自动化的前向传播、损失计算和参数更新，并返回损失值。
        此方法在每个 epoch 结束时会调用 `auto_step_scheduler` 来自动更新学习率调度器。
        Args:
            step_plateau_with_train_loss (bool): 传递给 `auto_step_scheduler`，
                若为 True，则强制使用训练损失更新 `ReduceLROnPlateau`。默认为 False。

        Returns:
            torch.Tensor: 计算出的损失张量。
        
        Raises:
            RuntimeError: 如果 `optimizer` 或 `criterion` 未在初始化时提供。
        """
        if self.optimizer is None or self.criterion is None:
            raise RuntimeError(
                "Cannot call auto_update() because either the optimizer or the "
                "criterion was not provided to the Trainer."
            )
        
        # 前向传播
        if isinstance(self.data, tuple):
            logits = self.model(*self.data)
        else:
            logits = self.model(self.data)
        # 计算损失
        loss = self.criterion(logits, self.target)
        # 反向传播和优化
        self.update(loss)
        # 调用专用的自动调度器步进方法
        self.auto_step_scheduler(loss, step_plateau_with_train_loss=step_plateau_with_train_loss)
        return loss

    def save_checkpoint(self, path: Optional[str] = None, extra_info: Optional[Dict[str, Any]] = None) -> None:
        """
        保存训练检查点。

        优先使用传入的 path，否则使用初始化时设置的默认路径。如果两者都未提供则报错。
        """
        path_to_use = path if path is not None else self.checkpoint_path
        if path_to_use is None:
            raise ValueError(
                "No checkpoint path provided. Please provide a path to save_checkpoint() "
                "or set a default checkpoint_path during Trainer initialization."
            )
        
        dir_path = os.path.dirname(path_to_use)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, path_to_use)

    def load_checkpoint(self, path: Optional[str] = None) -> 'Trainer':
        """
        加载训练检查点以恢复训练。

        优先使用传入的 path，否则使用初始化时设置的默认路径。如果两者都未提供则报错。
        """
        path_to_use = path if path is not None else self.checkpoint_path
        if path_to_use is None:
             raise ValueError(
                "No checkpoint path provided. Please provide a path to load_checkpoint() "
                "or set a default checkpoint_path during Trainer initialization."
            )

        if not os.path.exists(path_to_use):
            print(f"Warning: Checkpoint file not found at '{path_to_use}'. Starting from scratch.")
            return self

        try:
            checkpoint = torch.load(path_to_use, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.start_epoch = checkpoint.get('epoch', -1) + 1
            print(f"Checkpoint loaded from '{path_to_use}'. Resuming training from epoch {self.start_epoch}.")
        
        except Exception as e:
            print(f"Warning: Failed to load checkpoint from {path_to_use}. Starting from scratch. Error: {e}")
            self.start_epoch = 0
        
        return self
    
    def save_model(self, path: str) -> None:
        """
        保存训练模型。
        """
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """
        加载训练模型。
        """
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from '{path}'.")
        except Exception as e:
            print(f"Warning: Failed to load model from {path}. Error: {e}")
