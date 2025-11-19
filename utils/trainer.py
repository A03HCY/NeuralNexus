import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader
from typing import Any, Generator, Iterator, Optional, Dict, List
from tqdm import tqdm

def set_seed(seed: int = 42):
    """
    固定随机种子以保证实验的可复现性。

    Args:
        seed (int): 随机种子数值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 选择确定性算法 (可能会降低性能)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def match_shape_if_needed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    检查张量 a 是否需要调整形状以匹配 b（主要用于处理 [1, N] vs [N] 的情况）。

    Args:
        a (torch.Tensor): 预测张量或源张量。
        b (torch.Tensor): 目标张量。

    Returns:
        torch.Tensor: 调整形状后的张量 a。
    """
    if a.dim() == 2 and b.dim() == 1 and a.shape[0] == 1 and a.shape[1] == b.shape[0]:
        return a.squeeze(0)
    if a.dim() == 2 and b.dim() == 1 and a.shape[1] == 1 and a.shape[0] == b.shape[0]:
         return a.squeeze(1)
    return a

class Trainer:
    """
    一个增强版的全功能训练/评估迭代器。
    
    支持：
    - 自动混合精度训练 (AMP)
    - 梯度累积 (Gradient Accumulation)
    - 梯度裁剪 (Gradient Clipping)
    - 训练历史记录 (History Tracking)
    - 断点续训 (Resume Training)
    - 灵活的训练/评估循环 (Generator based loop)
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
        use_amp: bool = False,
        accumulation_steps: int = 1,
        grad_clip_norm: Optional[float] = None,
    ) -> None:
        """
        初始化 Trainer。

        Args:
            model: 待训练模型。
            num_epochs: 总 Epoch 数。
            train_loader: 训练数据加载器。
            test_loader: 测试数据加载器。
            optimizer: 优化器。
            criterion: 损失函数。
            scheduler: 学习率调度器。
            checkpoint_path: 检查点保存路径 (例如 'checkpoints/last.pt')。
            device: 运行设备 (默认自动检测)。
            use_amp: 是否开启自动混合精度训练 (需 GPU 支持)。
            accumulation_steps: 梯度累积步数 (默认为1，即不累积)。
            grad_clip_norm: 梯度裁剪的范数阈值 (None 表示不裁剪)。
        """
        self.model = model
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        
        # --- 高级训练配置 ---
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        self.accumulation_steps = accumulation_steps
        self.grad_clip_norm = grad_clip_norm

        # --- 设备管理 ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

        # --- 状态变量 ---
        self.epoch: int = 0
        self.batch_idx: int = 0
        self.global_step: int = 0  # 记录总的 batch 步数
        self.data: Optional[Any] = None
        self.target: Optional[Any] = None
        
        self.is_first_batch_in_epoch: bool = False
        self.is_last_batch_in_epoch: bool = False
        self.start_epoch: int = 0
        
        self.loss: Optional[torch.Tensor] = None # 当前 batch loss
        self.running_loss: float = 0.0 # 当前 epoch 累计 loss
        self.running_samples: int = 0  # 当前 epoch 累计样本数
        
        # --- 评估统计 ---
        self.eval_loss: float = 0.0
        self.correct_predictions: int = 0
        self.total_predictions: int = 0

        # --- 历史记录 ---
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        self.best_val_metric = -float('inf') # 用于保存 best model

        # --- 初始化 ---
        self._display_model_summary()
        if self.checkpoint_path is not None:
            self.load_checkpoint()

    @property
    def display_epoch(self) -> int:
        """获取当前用于显示的 Epoch 序号 (从1开始计数)。"""
        return max(self.epoch, self.start_epoch) + 1
    
    @property
    def epoch_mean_loss(self) -> float:
        """返回当前 Epoch 到目前为止的平均 Loss。"""
        if self.running_samples == 0:
            return 0.0
        return self.running_loss / self.running_samples
    
    @property
    def eval_accuracy(self) -> float:
        """返回当前评估阶段的累积准确率。"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def _display_model_summary(self):
        """打印模型结构、参数量及运行设备信息。"""
        print("-" * 60)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device} | AMP: {self.use_amp}")
        print(f"Gradient Accumulation: {self.accumulation_steps} steps")
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Params: {total / 1e6:.2f}M (Trainable: {trainable / 1e6:.2f}M)")
        print("-" * 60)

    def _process_batch_data(self, batch_data):
        """
        处理 Batch 数据，移动到计算设备并拆分 data/target。
        
        Args:
            batch_data: DataLoader 返回的一个 batch 数据。
        """
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) > 1:
                # 假设最后一个是 target，前面都是 input
                self.data = tuple(d.to(self.device) for d in batch_data[:-1])
                self.target = batch_data[-1].to(self.device)
                if len(self.data) == 1:
                    self.data = self.data[0]
            else:
                # 只有 data 没有 target (用于推理)
                self.data = batch_data[0].to(self.device)
                self.target = None
        else:
            # 假设 batch_data 本身就是 input，没有 target? 
            # 这里通常应该是 list/tuple，或者是字典，视 dataset 而定。
            # 为兼容旧逻辑：
            self.data = batch_data.to(self.device)
            self.target = None

    def _create_train_iterator(self, data_loader: DataLoader, tqdm_bar: bool, print_loss: bool) -> Generator['Trainer', None, None]:
        """
        内部生成器：执行训练循环逻辑。

        Args:
            data_loader: 数据加载器。
            tqdm_bar: 是否显示进度条。
            print_loss: Epoch 结束时是否打印 Loss。
        """
        self.model.train()
        num_batches = len(data_loader)
        
        for epoch_num in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch_num
            self.running_loss = 0.0
            self.running_samples = 0
            
            iterable = tqdm(data_loader, desc=f"Train Ep {self.display_epoch}/{self.num_epochs}", leave=False) if tqdm_bar else data_loader

            for batch_idx, batch_data in enumerate(iterable):
                self.batch_idx = batch_idx
                self.global_step += 1
                self._process_batch_data(batch_data)
                
                self.is_first_batch_in_epoch = (batch_idx == 0)
                self.is_last_batch_in_epoch = (batch_idx == num_batches - 1)
                
                yield self
            
            # Epoch 结束记录训练 Loss
            epoch_loss = self.epoch_mean_loss
            self.history['train_loss'].append(epoch_loss)
            
            if print_loss:
                print(f"Epoch {self.display_epoch} finished: Avg Loss = {epoch_loss:.4f}")

    def train(self, train_loader: Optional[DataLoader] = None, tqdm_bar: bool = True, print_loss: bool = True) -> Iterator['Trainer']:
        """
        创建训练迭代器。

        Args:
            train_loader (Optional[DataLoader]): 覆盖初始化的 train_loader。
            tqdm_bar (bool): 是否显示 tqdm 进度条。
            print_loss (bool): 是否在 Epoch 结束打印 Loss。

        Yields:
            Trainer: 返回 Trainer 实例本身，供外部循环调用 `auto_update` 等方法。
        """
        loader = train_loader if train_loader else self.train_loader
        if not loader:
            raise ValueError("No train_loader provided.")
        return self._create_train_iterator(loader, tqdm_bar, print_loss)

    def _create_eval_iterator(self, data_loader: DataLoader, description: str, tqdm_bar: bool) -> Generator['Trainer', None, None]:
        """
        内部生成器：执行评估/推理循环逻辑。

        Args:
            data_loader: 数据加载器。
            description: 进度条描述文本。
            tqdm_bar: 是否显示进度条。
        """
        self.model.eval()
        # 重置评估统计
        self.eval_loss = 0.0
        self.correct_predictions = 0
        self.total_predictions = 0
        
        try:
            iterable = tqdm(data_loader, desc=description, leave=False) if tqdm_bar else data_loader
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(iterable):
                    self.batch_idx = batch_idx
                    self.is_first_batch_in_epoch = (batch_idx == 0)
                    self._process_batch_data(batch_data)
                    yield self
        finally:
            self.model.train()

    def eval(self, test_loader: Optional[DataLoader] = None, description: str = "Evaluating", tqdm_bar: bool = True) -> Iterator['Trainer']:
        """
        创建评估迭代器。

        Args:
            test_loader (Optional[DataLoader]): 覆盖初始化的 test_loader。
            description (str): 进度条前缀描述。
            tqdm_bar (bool): 是否显示进度条。

        Yields:
            Trainer: 返回 Trainer 实例，供外部计算指标。
        """
        loader = test_loader if test_loader else self.test_loader
        if not loader:
            raise ValueError("No test_loader provided.")
        return self._create_eval_iterator(loader, description, tqdm_bar)

    def update(self, loss: torch.Tensor, step_plateau_with_train_loss: bool = False) -> None:
        """
        执行反向传播及参数更新。包含：梯度缩放(AMP)、梯度累积、梯度裁剪、优化器更新。

        Args:
            loss (torch.Tensor): 计算出的损失值。
            step_plateau_with_train_loss (bool): 若使用 ReduceLROnPlateau，是否使用训练 Loss 来更新调度器。
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not set.")

        # 1. 根据累积步数归一化 Loss
        loss = loss / self.accumulation_steps

        # 2. 反向传播 (AMP vs Normal)
        if self.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # 3. 仅在累积达到步数 或 Epoch 最后一个 Batch 时更新参数
        if (self.batch_idx + 1) % self.accumulation_steps == 0 or self.is_last_batch_in_epoch:
            
            # 梯度裁剪 (处理 AMP 需要先 unscale)
            if self.grad_clip_norm is not None:
                if self.use_amp and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            # 优化器步进
            if self.use_amp and self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.optimizer.zero_grad()
        
        # 更新调度器 (仅 epoch 结束)
        self.auto_step_scheduler(loss * self.accumulation_steps, step_plateau_with_train_loss)

    def auto_update(self, step_plateau_with_train_loss: bool = False) -> torch.Tensor:
        """
        自动执行完整训练步：Forward -> Loss -> Backward -> Update。

        Args:
            step_plateau_with_train_loss (bool): 是否使用 Train Loss 更新调度器。

        Returns:
            torch.Tensor: 当前 Batch 的 Loss 值。
        """
        if not self.optimizer or not self.criterion:
            raise RuntimeError("Optimizer or Criterion missing.")
        
        # 前向传播 (With AMP context support)
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            if isinstance(self.data, tuple):
                logits = self.model(*self.data)
            else:
                logits = self.model(self.data)
            
            loss = self.criterion(logits, self.target)
        
        self.loss = loss
        
        # 统计 Running Loss (还原 Batch Size 影响)
        batch_size = self.target.size(0) if hasattr(self.target, 'size') else 1
        self.running_loss += loss.item() * batch_size
        self.running_samples += batch_size

        self.update(loss, step_plateau_with_train_loss)
        return loss

    def auto_step_scheduler(self, loss_val: Optional[torch.Tensor] = None, use_train_loss: bool = False) -> None:
        """
        Epoch 结束时自动更新调度器。
        
        Args:
            loss_val: 当前的 Loss 值 (用于 ReduceLROnPlateau)。
            use_train_loss: 是否强制使用传入的 loss_val 更新调度器。
        """
        if not self.is_last_batch_in_epoch or self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            # 只有在明确要求使用 train loss 或者在纯训练循环中才使用 train loss 更新
            if use_train_loss and loss_val is not None:
                self.scheduler.step(loss_val.item())
            # 否则通常 ReduceLROnPlateau 是在 validation loop 后由用户手动调用 step(val_loss)
        else:
             # StepLR, CosineAnnealingLR 等通常不带 metric 参数
            self.scheduler.step()

    def calculate_classification_metrics(self) -> float:
        """
        计算简单的分类任务准确率指标，更新内部状态(eval_loss, correct_predictions)。

        Returns:
            float: 当前 Batch 的 Loss (scalar)。
        """
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            logits = self.model(self.data) if not isinstance(self.data, tuple) else self.model(*self.data)
            logits = match_shape_if_needed(logits, self.target)
            loss = self.criterion(logits, self.target) if self.criterion else torch.tensor(0.0)

        self.eval_loss += loss.item() * self.target.size(0)
        self.total_predictions += self.target.size(0)
        
        # 计算 Acc
        if logits.ndim > 1 and logits.shape[1] > 1:
            # 多分类
            preds = logits.argmax(dim=1)
            if self.target.ndim > 1: # target 也是 one-hot
                targets = self.target.argmax(dim=1)
            else:
                targets = self.target
            self.correct_predictions += (preds == targets).sum().item()
        else:
            # 二分类 (sigmoid)
            preds = (torch.sigmoid(logits) > 0.5).float()
            self.correct_predictions += (preds == self.target).sum().item()
            
        return loss.item()

    def record_history(self, current_val_loss: float = None, current_val_acc: float = None):
        """
        手动记录验证集指标到 history 字典中。

        Args:
            current_val_loss: 验证集 Loss。
            current_val_acc: 验证集 Accuracy。
        """
        if current_val_loss is not None:
            self.history['val_loss'].append(current_val_loss)
        if current_val_acc is not None:
            self.history['val_acc'].append(current_val_acc)

    def auto_checkpoint(self, metrics: Optional[Dict[str, float]] = None, save_best_only: bool = False, monitor: str = 'val_acc') -> None:
        """
        自动保存检查点。根据 monitor 指标自动判断是否保存为最佳模型。

        Args:
            metrics: 当前 epoch 的指标字典，用于判断是否是最佳模型 (例如 {'val_loss': 0.5, 'val_acc': 0.9})。
            save_best_only: (该参数暂未在逻辑中完全隔离，目前逻辑是同时保存 last 和 best)。
            monitor: 监控哪个指标来决定 best model (例如 'val_acc' 或 'val_loss')。
        """
        if not self.is_last_batch_in_epoch or not self.checkpoint_path:
            return
        
        is_best = False
        if metrics and monitor in metrics:
            current_val = metrics[monitor]
            # 简单的 best 逻辑：loss越小越好，acc越大越好
            if 'loss' in monitor:
                if self.best_val_metric == -float('inf'): self.best_val_metric = float('inf')
                if current_val < self.best_val_metric:
                    self.best_val_metric = current_val
                    is_best = True
            else:
                if current_val > self.best_val_metric:
                    self.best_val_metric = current_val
                    is_best = True
        
        # 保存最新
        self.save_checkpoint(extra_info=metrics)
        # 保存最佳
        if is_best:
            best_path = os.path.join(os.path.dirname(self.checkpoint_path), 'best_model.pt')
            self.save_checkpoint(path=best_path, extra_info=metrics)
            print(f" -> New best model saved at epoch {self.display_epoch} ({monitor}: {metrics[monitor]:.4f})")

    def save_checkpoint(self, path: Optional[str] = None, extra_info: Optional[Dict[str, Any]] = None) -> None:
        """
        保存模型检查点。

        Args:
            path: 保存路径 (默认使用初始化时的 checkpoint_path)。
            extra_info: 需要额外保存的字典信息。
        """
        path_to_use = path if path is not None else self.checkpoint_path
        if path_to_use is None: return
        
        os.makedirs(os.path.dirname(path_to_use), exist_ok=True)
        
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'history': self.history,
            'best_val_metric': self.best_val_metric
        }
        if extra_info:
            state.update(extra_info)
        
        try:
            torch.save(state, path_to_use)
        except Exception as e:
            print(f"Error saving checkpoint {path_to_use}: {e}")

    def load_checkpoint(self, path: Optional[str] = None) -> 'Trainer':
        """
        加载模型检查点以恢复训练。

        Args:
            path: 检查点路径 (默认使用初始化时的 checkpoint_path)。

        Returns:
            Trainer: 返回自身实例。
        """
        path_to_use = path if path is not None else self.checkpoint_path
        if path_to_use is None or not os.path.exists(path_to_use):
            return self

        print(f"Loading checkpoint: {path_to_use}")
        try:
            checkpoint = torch.load(path_to_use, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and checkpoint.get('optimizer_state_dict'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            self.start_epoch = checkpoint.get('epoch', -1) + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.history = checkpoint.get('history', self.history)
            self.best_val_metric = checkpoint.get('best_val_metric', -float('inf'))
            
            print(f"Resumed from Epoch {self.start_epoch}.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")
            self.start_epoch = 0
        
        return self
    
    def save_model(self, path: str) -> None:
        """
        仅保存模型的权重参数 (state_dict)。
        通常用于推理部署，文件体积比 checkpoint 小。

        Args:
            path (str): 保存路径 (例如 'models/resnet_weights.pth')。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            # 如果模型被 DataParallel 包装，建议保存 model.module.state_dict()
            # 这里为了通用性，直接保存 model.state_dict()
            torch.save(self.model.state_dict(), path)
            print(f"Model weights saved to: {path}")
        except Exception as e:
            print(f"Error saving model weights to {path}: {e}")

    def load_model(self, path: str, strict: bool = True) -> None:
        """
        加载模型权重。自动处理“纯权重文件”和“完整检查点文件”。

        Args:
            path (str): 权重文件路径。
            strict (bool): 是否严格匹配键值 (默认 True)。
                           如果做迁移学习(修改了网络层)，可设为 False 以忽略不匹配的键。
        """
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path}")
            return

        print(f"Loading model weights from: {path}")
        try:
            state_dict = torch.load(path, map_location=self.device)
            
            # 兼容性处理：如果传入的是完整 checkpoint 字典，则提取 model_state_dict
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                print("Detected full checkpoint, extracting 'model_state_dict'...")
                state_dict = state_dict['model_state_dict']
            
            # 加载权重
            missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)
            
            if not strict:
                if missing:
                    print(f"Missing keys (ignored): {len(missing)} keys")
                if unexpected:
                    print(f"Unexpected keys (ignored): {len(unexpected)} keys")
            
            print("Model weights loaded successfully.")
            
        except Exception as e:
            print(f"Failed to load model weights: {e}")
