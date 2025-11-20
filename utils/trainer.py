import os
import copy
import random
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader
from typing import Any, Generator, Iterator, Optional, Dict, List, Union, Tuple
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    绘制混淆矩阵并返回 matplotlib figure 对象。
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_score, class_names=None):
    """
    绘制 ROC 曲线并返回 matplotlib figure 对象。
    支持二分类和多分类 (One-vs-Rest)。
    """
    n_classes = len(class_names) if class_names else (y_score.shape[1] if y_score.ndim > 1 else 2)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 二分类情况 (y_score shape [N, 1] or [N])
    if n_classes == 2 or (y_score.ndim == 1) or (y_score.shape[1] == 1):
        # 假设 y_score 是正类的概率
        # 如果 y_score 是 [N, 1]，squeeze
        if y_score.ndim == 2: y_score = y_score.squeeze()
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    else:
        # 多分类情况
        # 需要将 y_true 二值化
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # 计算每一类的 ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        # 绘制微平均 ROC 曲线 (Micro-average)
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        # 绘制每一类的 ROC 曲线
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
        for i, color in zip(range(n_classes), colors):
            label = f'ROC curve of class {class_names[i] if class_names else i}'
            label += f' (area = {roc_auc[i]:.2f})'
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig

def check_sanity(loss: torch.Tensor, step: int):
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"CRITICAL WARNING: Loss became NaN or Inf at step {step}!")
        return False
    return True

def set_seed(seed: int = 42):
    """
    固定随机种子以保证实验的可复现性。
    
    包括 random, numpy, torch 以及 cuda 的种子设置。
    注意：设置 cudnn.deterministic = True 可能会降低训练速度。

    Args:
        seed (int): 随机种子数值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 选择确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def match_shape_if_needed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    检查张量 a 是否需要调整形状以匹配 b。
    
    常用于处理 Binary Cross Entropy 中预测值 [N, 1] 与 目标值 [N] 不匹配的情况。

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

class TimingContext:
    def __init__(self, name="Block"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        torch.cuda.synchronize() # 如果用 GPU，必须同步才能测准
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        print(f"[{self.name}] elapsed: {(time.time() - self.start)*1000:.2f} ms")

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.epoch_start_time = 0
    
    def start_epoch(self):
        self.epoch_start_time = time.time()
        
    def end_epoch(self) -> str:
        elapsed = time.time() - self.epoch_start_time
        return str(datetime.timedelta(seconds=int(elapsed)))
    
    def total_time(self) -> str:
        elapsed = time.time() - self.start_time
        return str(datetime.timedelta(seconds=int(elapsed)))

class ModelEMA:
    """
    模型权重的指数移动平均 (Exponential Moving Average)。
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # 创建影子模型 (Shadow Model)，只保存权重，不参与反向传播
        self.shadow = copy.deepcopy(model.module if hasattr(model, 'module') else model)
        self.shadow.eval()
        # 将参数设为不需要梯度，节省显存
        for param in self.shadow.parameters():
            param.requires_grad = False
    def update(self, model: nn.Module):
        """根据当前模型参数更新影子参数"""
        # 兼容 DataParallel
        model_to_use = model.module if hasattr(model, 'module') else model
        
        with torch.no_grad():
            msd = model_to_use.state_dict()
            for name, param in self.shadow.named_parameters():
                if name in msd:
                    new_param = msd[name]
                    # 确保设备一致
                    if param.device != new_param.device:
                         param.data = param.data.to(new_param.device)
                    # EMA 公式: shadow = decay * shadow + (1 - decay) * new_param
                    param.data.mul_(self.decay).add_(new_param.data, alpha=1 - self.decay)
    def apply_shadow(self, model: nn.Module):
        """将 EMA 权重赋值给原模型 (通常在推理或保存最佳模型前使用)"""
        model_to_use = model.module if hasattr(model, 'module') else model
        model_to_use.load_state_dict(self.shadow.state_dict())

class Trainer:
    """
    一个全功能的 PyTorch 训练/评估迭代器封装类。
    该类旨在通过生成器模式 (Generator Pattern) 简化训练循环，同时保留极高的灵活性。
    
    主要特性:
    - **自动混合精度 (AMP)**: 支持 fp16 训练。
    - **梯度策略**: 支持梯度累积 (Gradient Accumulation) 和梯度裁剪 (Gradient Clipping)。
    - **生命周期管理**: 支持断点续训 (Resume)、模型保存 (Checkpointing)、早停 (Early Stopping)。
    - **可视化**: 集成 TensorBoard 日志记录。
    - **易用性**: 自动处理设备移动、进度条显示 (tqdm) 和指标计算。
    - **高级功能**: 支持 EMA、DataParallel、推理预测 (Predict)。
    Attributes:
        model (nn.Module): 正在训练的模型。
        device (torch.device): 当前运行设备。
        history (Dict): 记录训练过程中的 loss 和 metric 历史。
        global_step (int): 全局训练步数计数器。
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
        use_ema: bool = False,
        ema_decay: float = 0.999
    ) -> None:
        """
        初始化 Trainer 实例。

        Args:
            model (nn.Module): 待训练的 PyTorch 模型。
            num_epochs (int): 训练的总 Epoch 数。
            train_loader (DataLoader, optional): 训练数据加载器。
            test_loader (DataLoader, optional): 验证/测试数据加载器。
            optimizer (Optimizer, optional): 优化器实例。
            criterion (nn.Module, optional): 损失函数实例。
            scheduler (_LRScheduler, optional): 学习率调度器。
            checkpoint_path (str, optional): 检查点保存路径 (例如 'checkpoints/ckpt.pt')。
            device (torch.device, optional): 指定运行设备。若为 None 则自动检测 CUDA/CPU。
            use_amp (bool): 是否开启自动混合精度训练 (需要 GPU 支持)。
            accumulation_steps (int): 梯度累积步数，默认为 1 (不累积)。
            grad_clip_norm (float, optional): 梯度裁剪的最大范数。None 表示不裁剪。
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

        # --- 设备管理与 DataParallel ---
        # 逻辑顺序：确定 Device -> 移动模型 -> 包装 DataParallel
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # 先将模型移动到指定设备 (这对 DataParallel 很重要)
        self.model.to(self.device)
        
        # 自动检测并应用 DP (无论 device 是手动传入还是自动检测)
        self.use_dp = False
        self._try_init_dataparallel()

        # --- 初始化 EMA ---
        self.use_ema = use_ema
        self.ema = ModelEMA(self.model, decay=ema_decay) if use_ema else None
        if self.use_ema:
            print(f"EMA enabled with decay {ema_decay}")

        # --- 状态变量 ---
        self.epoch: int = 0
        self.start_epoch: int = 0
        self.batch_idx: int = 0
        self.global_step: int = 0  # 记录总的 optimizer step 次数 (batch数)
        
        # 当前 Batch 的数据
        self.data: Optional[Union[torch.Tensor, Tuple[torch.Tensor]]] = None
        self.target: Optional[torch.Tensor] = None
        
        # 循环控制标志
        self.is_first_batch_in_epoch: bool = False
        self.is_last_batch_in_epoch: bool = False
        
        # 统计变量
        self.loss: Optional[torch.Tensor] = None # 最近一次 forward 的 loss
        self.running_loss: float = 0.0 # 当前 epoch 累计 loss
        self.running_samples: int = 0  # 当前 epoch 累计样本数
        
        # --- 评估统计 ---
        self.eval_loss: float = 0.0
        self.correct_predictions: int = 0
        self.total_predictions: int = 0
        self.timer = Timer()

        # --- 历史记录 ---
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        self.best_val_metric = -float('inf') # 用于保存 best model

        # --- TensorBoard & Early Stopping ---
        self.writer = None  # TensorBoard writer
        self.patience_counter: int = 0 # 早停计数器
        self.best_metric_for_es: Optional[float] = None # 用于早停的最佳指标

        # --- 混淆矩阵相关 ---
        self.classes: Optional[Union[List, Tuple]] = None
        self.top_k: Optional[int] = None
        self.y_trues: List[torch.Tensor] = []
        self.y_preds: List[torch.Tensor] = []
        self.y_scores: List[torch.Tensor] = [] # 用于 ROC 曲线
        self.correct_top_k_predictions: int = 0
        self.enable_confusion_matrix: bool = False
        self.enable_roc_curve: bool = False

        # --- 初始化 ---
        self._display_model_summary()
        if self.checkpoint_path is not None:
            # 尝试自动加载 'last.pt' 或指定路径
            self.load_checkpoint()

    def init_classes(self, classes: Union[List, Tuple], top_k: Optional[int] = None, 
                     force_confusion_matrix: bool = False, force_roc_curve: bool = False):
        """
        初始化类别名称列表，用于绘制混淆矩阵和 ROC 曲线。
        
        Args:
            classes: 类别名称列表。
            top_k: 如果指定，将在评估时计算 Top-k Accuracy。
            force_confusion_matrix: 是否强制生成混淆矩阵 (即使类别数很多)。
            force_roc_curve: 是否强制生成 ROC 曲线 (即使类别数很多)。
        """
        self.classes = classes
        self.top_k = top_k
        
        num_classes = len(classes)
        
        # 智能判断是否开启混淆矩阵 (默认阈值 50)
        if num_classes <= 50 or force_confusion_matrix:
            self.enable_confusion_matrix = True
        else:
            self.enable_confusion_matrix = False
            print(f"Confusion Matrix disabled due to large number of classes ({num_classes} > 50). Use force_confusion_matrix=True to override.")

        # 智能判断是否开启 ROC 曲线 (默认阈值 10)
        if num_classes <= 10 or force_roc_curve:
            self.enable_roc_curve = True
        else:
            self.enable_roc_curve = False
            print(f"ROC Curve disabled due to large number of classes ({num_classes} > 10). Use force_roc_curve=True to override.")

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
    
    def _try_init_dataparallel(self):
        """尝试初始化 DataParallel"""
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
            self.model = nn.DataParallel(self.model)
            self.use_dp = True
        else:
            self.use_dp = False

    def _display_model_summary(self):
        """
        打印丰富的模型结构、环境信息及参数统计。
        支持自动调用 torchinfo (如果已安装)。
        """
        import sys
        
        # 1. 获取实际模型 (处理 DataParallel)
        real_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 2. 基础统计
        total_params = sum(p.numel() for p in real_model.parameters())
        trainable_params = sum(p.numel() for p in real_model.parameters() if p.requires_grad)
        # 估算模型权重占用的显存 (Float32 = 4 bytes)
        # 注意：这只是静态权重，不包含中间激活值和梯度
        param_memory_mb = total_params * 4 / (1024 ** 2) 
        
        # 3. 格式化打印
        print("=" * 80)
        print(f"🟢 SYSTEM & ENV SUMMARY")
        print("-" * 80)
        print(f"{'PyTorch Version':<20} : {torch.__version__}")
        print(f"{'Python Version':<20} : {sys.version.split()[0]}")
        print(f"{'Device':<20} : {self.device}")
        
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            print(f"{'GPU Name':<20} : {gpu_name}")
            print(f"{'CUDA Version':<20} : {torch.version.cuda}")
            if hasattr(self, 'use_dp') and self.use_dp:
                 print(f"{'Distributed':<20} : DataParallel (GPUs: {torch.cuda.device_count()})")
        
        print("-" * 80)
        print(f"🔵 TRAINING CONFIG")
        print("-" * 80)
        print(f"{'AMP (Mixed Precision)':<25} : {'ON' if self.use_amp else 'OFF'}")
        print(f"{'Gradient Accumulation':<25} : {self.accumulation_steps} steps")
        print(f"{'Gradient Clipping':<25} : {self.grad_clip_norm if self.grad_clip_norm else 'OFF'}")
        print(f"{'Optimizer':<25} : {self.optimizer.__class__.__name__ if self.optimizer else 'None'}")
        if self.optimizer:
            try:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"{'Initial Learning Rate':<25} : {lr}")
            except: pass
        print(f"{'EMA (Exp Moving Avg)':<25} : {'ON' if (hasattr(self, 'use_ema') and self.use_ema) else 'OFF'}")

        print("-" * 80)
        print(f"🟡 MODEL SUMMARY: {real_model.__class__.__name__}")
        print("-" * 80)
        print(f"{'Layer (type)':<30} | {'Params':>12} | {'Trainable':>10}")
        print("-" * 60)
        
        for name, module in real_model.named_children():
            # 计算子模块参数
            mod_params = sum(p.numel() for p in module.parameters())
            mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            is_train = "Yes" if mod_trainable > 0 else "No"
            name_str = f"{name} ({module.__class__.__name__})"
            # 截断过长的名字
            if len(name_str) > 28: name_str = name_str[:25] + "..."
            
            print(f"{name_str:<30} | {mod_params:>12,} | {is_train:>10}")
        
        print("-" * 60)
        print(f"{'Total Params':<30} : {total_params:,}")
        print(f"{'Trainable Params':<30} : {trainable_params:,} ({trainable_params/total_params:.1%})")
        print(f"{'Non-Trainable Params':<30} : {total_params - trainable_params:,}")
        print(f"{'Est. Model Size (Weights)':<30} : {param_memory_mb:.2f} MB")
        
        print("=" * 80)

    def init_tensorboard(self, log_dir: str = "runs") -> 'Trainer':
        """
        初始化 TensorBoard SummaryWriter。
        如果 torch.utils.tensorboard 未安装，则不做任何操作。

        Args:
            log_dir (str): 日志保存目录。
        
        Returns:
            self (Trainer): 返回当前 Trainer 对象，以便链式调用。
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            # 获取模型名称
            real_model = self.model.module if hasattr(self.model, 'module') else self.model
            model_name = real_model.__class__.__name__
            
            # 如果用户使用默认路径，自动添加模型名和时间戳
            if log_dir == "runs":
                timestamp = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
                log_dir = os.path.join("runs", f"{model_name}_{timestamp}")
            
            self.writer = SummaryWriter(log_dir=log_dir)
            
            # 记录模型名称
            self.writer.add_text("Model/Name", model_name)
            
            print(f"TensorBoard initialized. Logs will be saved to: {log_dir}")
        except ImportError:
            print("Warning: TensorBoard not found. Install it using 'pip install tensorboard'.")
        finally:
            return self

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        手动记录指标到 TensorBoard。

        Args:
            metrics (Dict[str, float]): 指标字典，如 {'Val/Loss': 0.5, 'Val/Acc': 0.9}。
            step (int, optional): 当前步数。如果不填，默认使用 self.global_step。
        """
        if self.writer is None:
            return
        
        step_to_use = step if step is not None else self.global_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step_to_use)
    
    def log_confusion_matrix(self, loader, class_names=None):
        if self.writer is None: return
        preds, targets = self.predict(loader, return_targets=True)
        # 转换为类别索引
        if preds.ndim > 1: preds = preds.argmax(dim=1)
        if targets.ndim > 1: targets = targets.argmax(dim=1)
        
        fig = plot_confusion_matrix(targets.numpy(), preds.numpy(), class_names)
        self.writer.add_figure("Eval/Confusion_Matrix", fig, self.global_step)
        plt.close(fig)

    def check_early_stopping(self, current_metric: float, monitor: str = 'val_loss', patience: int = 5) -> bool:
        """
        检查是否触发早停 (Early Stopping)。

        Args:
            current_metric (float): 当前 epoch 的验证指标值。
            monitor (str): 监控指标名称 ('val_loss' 或 'val_acc')，用于决定是 'min' 还是 'max' 模式。
                           包含 'loss' 视为越小越好，否则视为越大越好。
            patience (int): 容忍多少个 epoch 指标未改善。

        Returns:
            bool: 如果返回 True，则应当停止训练循环。
        """
        # 首次调用初始化
        if self.best_metric_for_es is None:
             self.best_metric_for_es = float('inf') if 'loss' in monitor.lower() else -float('inf')

        is_better = False
        if 'loss' in monitor.lower():
            if current_metric < self.best_metric_for_es:
                is_better = True
        else:
            if current_metric > self.best_metric_for_es:
                is_better = True

        if is_better:
            self.best_metric_for_es = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            # 只有在计数器增加时才打印
            if self.patience_counter > 0:
                print(f"Early Stopping Counter: {self.patience_counter}/{patience}")

        if self.patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            return True
        
        return False

    def find_lr(self, train_loader: DataLoader = None, init_value: float = 1e-8, final_value: float = 10.0, beta: float = 0.98) -> None:
        """
        模拟训练以寻找最佳学习率。会绘制 Loss vs LR 曲线并保存到本地。
        注意：运行此方法后会重置模型参数到运行前状态。
        """
        train_loader = train_loader if train_loader else self.train_loader
        if train_loader is None:
            raise ValueError("No train_loader provided.")
        print("Finding learning rate...")
        # 1. 保存当前状态以恢复
        if isinstance(self.model, nn.DataParallel):
            model_state = copy.deepcopy(self.model.module.state_dict())
        else:
            model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        
        self.model.train()
        num = len(train_loader) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        
        avg_loss = 0.0
        best_loss = 0.0
        batch_num = 0
        losses = []
        lrs = []
        
        # 禁用 AMP scaler 避免干扰，或者创建一个临时的
        scaler = GradScaler() if self.use_amp else None
        
        try:
            for batch_data in tqdm(train_loader, desc="LR Finder", leave=False):
                batch_num += 1
                self._process_batch_data(batch_data)
                
                # Forward
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    if isinstance(self.data, tuple):
                        logits = self.model(*self.data)
                    else:
                        logits = self.model(self.data)
                    loss = self.criterion(logits, self.target)
                
                # Compute the smoothed loss
                loss_val = loss.item()
                avg_loss = beta * avg_loss + (1 - beta) * loss_val
                smoothed_loss = avg_loss / (1 - beta**batch_num)
                
                # Stop if the loss is exploding
                if batch_num > 1 and smoothed_loss > 4 * best_loss:
                    break
                if smoothed_loss < best_loss or batch_num == 1:
                    best_loss = smoothed_loss
                
                losses.append(smoothed_loss)
                lrs.append(lr)
                
                # Optimize
                self.optimizer.zero_grad()
                if self.use_amp and scaler:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # Update LR
                lr *= mult
                self.optimizer.param_groups[0]['lr'] = lr
        finally:
            # 2. 恢复模型状态
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(model_state)
            else:
                self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            print("LR Finder finished. Model state restored.")

        # 3. 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('lr_finder_result.png')
        print("Result saved to 'lr_finder_result.png'.")

    def _process_batch_data(self, batch_data: Any):
        """
        内部方法：处理 Batch 数据，将其移动到计算设备并拆分为 data 和 target。
        
        Args:
            batch_data: DataLoader 返回的一个 batch 数据。
        """
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) > 1:
                # 假设最后一个是 target，前面都是 input
                # 处理多输入的情况
                inputs = tuple(d.to(self.device) for d in batch_data[:-1])
                self.target = batch_data[-1].to(self.device)
                
                # 如果只有一个输入，解包 tuple
                if len(inputs) == 1:
                    self.data = inputs[0]
                else:
                    self.data = inputs
            else:
                # 只有数据没有标签（如无监督学习）
                self.data = batch_data[0].to(self.device)
                self.target = None
        else:
            # 只有 tensor
            self.data = batch_data.to(self.device)
            self.target = None

    def _create_train_iterator(self, data_loader: DataLoader, tqdm_bar: bool, print_loss: bool) -> Generator['Trainer', None, None]:
        """内部方法：生成训练循环的迭代器。"""
        self.model.train()
        num_batches = len(data_loader)
        
        for epoch_num in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch_num
            self.running_loss = 0.0
            self.running_samples = 0
            
            iterable = tqdm(data_loader, desc=f"Train Ep {self.display_epoch}/{self.num_epochs}", leave=False) if tqdm_bar else data_loader

            self.timer.start_epoch()
            for batch_idx, batch_data in enumerate(iterable):
                self.batch_idx = batch_idx
                self.global_step += 1
                self._process_batch_data(batch_data)
                
                self.is_first_batch_in_epoch = (batch_idx == 0)
                self.is_last_batch_in_epoch = (batch_idx == num_batches - 1)
                
                # Yield self allowing external control loop
                yield self
            
            # Epoch 结束记录
            epoch_time = self.timer.end_epoch()
            epoch_loss = self.epoch_mean_loss
            self.history['train_loss'].append(epoch_loss)
            
            # Log epoch loss to TensorBoard
            self.log({'Train/Epoch_Loss': epoch_loss}, step=self.display_epoch)
            
            if print_loss:
                print(f"Epoch {self.display_epoch} finished in {epoch_time}. Avg Loss = {epoch_loss:.6f}")
            
            # 可以在这里加入 scheduler step (epoch级)
            if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                # 简单的 epoch step，如果需要 metric step 需在外部调用 auto_step_scheduler
                if not hasattr(self.scheduler, 'step_batch'): # 排除 warmup 等 batch 级 scheduler
                     self.scheduler.step()

    def train(self, train_loader: Optional[DataLoader] = None, tqdm_bar: bool = True, print_loss: bool = True) -> Iterator['Trainer']:
        """
        创建训练迭代器。
        
        使用方法:
            for trainer in trainer.train():
                loss = trainer.auto_update()
                或者自定义 update 逻辑

        Args:
            train_loader (DataLoader, optional): 覆盖初始化的 DataLoader。
            tqdm_bar (bool): 是否显示进度条。
            print_loss (bool): 是否在 Epoch 结束时打印平均 Loss。

        Returns:
            Iterator['Trainer']: 产生 Trainer 实例的生成器。
        """
        loader = train_loader if train_loader else self.train_loader
        if not loader:
            raise ValueError("No train_loader provided.")
        return self._create_train_iterator(loader, tqdm_bar, print_loss)

    def _create_eval_iterator(self, data_loader: DataLoader, description: str, tqdm_bar: bool) -> Generator['Trainer', None, None]:
        """内部方法：生成评估循环的迭代器。"""
        self.model.eval()
        self.eval_loss = 0.0
        self.correct_predictions = 0
        self.correct_top_k_predictions = 0
        self.total_predictions = 0
        num_batches = len(data_loader)
        
        try:
            iterable = tqdm(data_loader, desc=description, leave=False) if tqdm_bar else data_loader
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(iterable):
                    self.batch_idx = batch_idx
                    self.is_first_batch_in_epoch = (batch_idx == 0)
                    self.is_last_batch_in_epoch = (batch_idx == num_batches - 1)
                    self._process_batch_data(batch_data)
                    yield self
        finally:
            # 恢复训练模式
            self.model.train()

    def eval(self, test_loader: Optional[DataLoader] = None, description: str = "Evaluating", tqdm_bar: bool = True) -> Iterator['Trainer']:
        """
        创建评估迭代器。

        使用方法:
            for trainer in trainer.eval():
                trainer.calculate_classification_metrics()

        Args:
            test_loader (DataLoader, optional): 覆盖初始化的 DataLoader。
            description (str): 进度条描述文字。
            tqdm_bar (bool): 是否显示进度条。

        Returns:
            Iterator['Trainer']: 产生 Trainer 实例的生成器。
        """
        loader = test_loader if test_loader else self.test_loader
        if not loader:
            raise ValueError("No test_loader provided.")
        return self._create_eval_iterator(loader, description, tqdm_bar)

    def update(self, loss: torch.Tensor, step_plateau_with_train_loss: bool = False) -> None:
        """
        执行反向传播及参数更新的核心逻辑。
        
        包含：梯度缩放 (AMP)、梯度累积、梯度裁剪、优化器更新。

        Args:
            loss (torch.Tensor): 计算出的损失值。
            step_plateau_with_train_loss (bool): 是否使用训练 Loss 更新 ReduceLROnPlateau 调度器。
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not set.")
        loss = loss / self.accumulation_steps
        if self.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if (self.global_step % self.accumulation_steps == 0) or self.is_last_batch_in_epoch:
            
            # AMP Unscale (为了能够正确计算梯度范数和裁剪)
            if self.use_amp and self.scaler:
                self.scaler.unscale_(self.optimizer)
            # 记录梯度范数 (Gradient Norm) ---
            # 如果启用了裁剪，clip_grad_norm_ 会返回原始范数；
            # 如果未启用裁剪，我们需要手动计算范数用于记录。
            grad_norm = 0.0
            if self.grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                grad_norm = grad_norm.item()
            else:
                # 手动计算范数用于日志 (不修改梯度)
                parameters = [p for p in self.model.parameters() if p.grad is not None]
                if parameters:
                    device = parameters[0].grad.device
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
                    grad_norm = total_norm.item()
            
            # 记录到 TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Train/Grad_Norm', grad_norm, self.global_step)
            # 优化器步进
            if self.use_amp and self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            
            # 更新 EMA
            if self.use_ema and self.ema is not None:
                self.ema.update(self.model)
        
        self.auto_step_scheduler(loss * self.accumulation_steps, step_plateau_with_train_loss)

    def auto_update(self, step_plateau_with_train_loss: bool = False) -> torch.Tensor:
        """
        自动执行完整训练步：Forward -> Loss -> Backward -> Update。
        
        如果 TensorBoard 已启用，会自动记录 Batch Loss 和学习率。

        Args:
            step_plateau_with_train_loss (bool): 传递给 update 方法。

        Returns:
            torch.Tensor: 当前 batch 的原始 Loss 值 (未经过 accumulate 缩放)。
        """
        if not self.optimizer or not self.criterion:
            raise RuntimeError("Optimizer or Criterion missing.")
        
        # Forward & Loss
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            if isinstance(self.data, tuple):
                logits = self.model(*self.data)
            else:
                logits = self.model(self.data)
            
            loss = self.criterion(logits, self.target)

        if not check_sanity(loss, self.global_step):
            raise ValueError("Loss became NaN, stopping training.")
        if loss.ndim > 0:
            loss = loss.mean()
        self.loss = loss
        
        # 统计 Running Loss
        batch_size = self.target.size(0) if hasattr(self.target, 'size') else 1
        loss_scalar = loss.item()
        self.running_loss += loss_scalar * batch_size
        self.running_samples += batch_size

        # 自动记录 TensorBoard (Batch级)
        if self.writer is not None:
            self.writer.add_scalar('Train/Batch_Loss', loss_scalar, self.global_step)
            # 记录学习率 (取第一个 param_group)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LR', current_lr, self.global_step)

        # Backward & Update
        self.update(loss, step_plateau_with_train_loss)
        return loss

    def auto_step_scheduler(self, loss_val: Optional[torch.Tensor] = None, use_train_loss: bool = False) -> None:
        """
        辅助方法：根据调度器类型自动执行 step。
        主要用于处理 ReduceLROnPlateau 需要 metric 的情况。
        """
        if self.scheduler is None:
            return

        # 如果是 Plateau 调度器，通常在 Epoch 结束时调用，但如果用户希望基于 batch loss 也可以
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if use_train_loss and loss_val is not None and self.is_last_batch_in_epoch:
                self.scheduler.step(loss_val.item())
        # 如果是 OneCycleLR 或其他需要每个 batch step 的调度器
        elif hasattr(self.scheduler, 'step_batch'): # 自定义属性标记或检查类型
             pass # 通常由外部显式调用，或者在这里添加逻辑

    def calculate_classification_metrics(self) -> float:
        """
        计算常规分类任务的 Loss 和 Accuracy。
        更新 eval_loss 和 correct_predictions。
        如果调用了 init_classes，还会累积预测结果并在 epoch 结束时绘制混淆矩阵。

        Returns:
            float: 当前 Batch 的 Loss。
        """
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            if isinstance(self.data, tuple):
                logits = self.model(*self.data)
            else:
                logits = self.model(self.data)
            
            # 处理 shape [N,1] vs [N]
            logits_squeezed = match_shape_if_needed(logits, self.target)
            loss = self.criterion(logits_squeezed, self.target) if self.criterion else torch.tensor(0.0)

        batch_size = self.target.size(0)
        self.eval_loss += loss.item() * batch_size
        self.total_predictions += batch_size
        
        # 计算 Acc 并准备混淆矩阵数据
        preds = None
        targets = None
        scores = None # 用于 ROC

        # 多分类 (Logits shape [N, C], C > 1)
        if logits.ndim > 1 and logits.shape[1] > 1:
            preds = logits.argmax(dim=1)
            scores = torch.softmax(logits, dim=1) # 概率
            if self.target.ndim > 1: # target 是 one-hot 或 probabilities
                targets = self.target.argmax(dim=1)
            else: # target 是 indices
                targets = self.target
            self.correct_predictions += (preds == targets).sum().item()
        # 二分类 (Logits shape [N, 1] 或 [N])
        else:
            # 假设 logits 为 raw score，应用 sigmoid
            if logits_squeezed.ndim == 0: # scalar
                 scores = torch.sigmoid(logits_squeezed)
                 preds = (scores > 0.5).float()
            else:
                 scores = torch.sigmoid(logits_squeezed)
                 preds = (scores > 0.5).float()
            targets = self.target
            self.correct_predictions += (preds == targets).sum().item()
        
        # 计算 Top-k Accuracy
        if self.top_k is not None and logits.ndim > 1 and logits.shape[1] >= self.top_k:
            # logits: [N, C], target: [N]
            _, pred_topk = logits.topk(self.top_k, dim=1, largest=True, sorted=True) # [N, k]
            pred_topk = pred_topk.t() # [k, N]
            correct = pred_topk.eq(targets.view(1, -1).expand_as(pred_topk))
            self.correct_top_k_predictions += correct.reshape(-1).float().sum().item()

        # 累积混淆矩阵和 ROC 数据
        if self.classes is not None:
            if self.enable_confusion_matrix:
                self.y_preds.append(preds.detach().cpu())
                self.y_trues.append(targets.detach().cpu())
            
            if self.enable_roc_curve:
                # 确保 y_trues 也被收集 (如果上面没收集)
                if not self.enable_confusion_matrix:
                    self.y_trues.append(targets.detach().cpu())
                self.y_scores.append(scores.detach().cpu())

            # 如果是最后一个 batch，生成图表
            if self.is_last_batch_in_epoch:
                all_trues = torch.cat(self.y_trues).numpy()
                
                # 1. 混淆矩阵
                if self.enable_confusion_matrix:
                    all_preds = torch.cat(self.y_preds).numpy()
                    fig_cm = plot_confusion_matrix(all_trues, all_preds, self.classes)
                    if self.writer is not None:
                        self.writer.add_figure("Eval/Confusion_Matrix", fig_cm, self.global_step)
                    else:
                        save_path = f"confusion_matrix_epoch_{self.display_epoch}.png"
                        fig_cm.savefig(save_path)
                        print(f"Confusion matrix saved to {save_path}")
                    plt.close(fig_cm)

                    # 打印 Per-Class Accuracy
                    cm = confusion_matrix(all_trues, all_preds)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        per_class_acc = cm.diagonal() / cm.sum(axis=1)
                        per_class_acc = np.nan_to_num(per_class_acc)
                    
                    print("-" * 40)
                    print(f"{'Class':<15} | {'Accuracy':<10}")
                    print("-" * 40)
                    for i, acc in enumerate(per_class_acc):
                        if i < len(self.classes):
                            class_name = str(self.classes[i])
                            print(f"Accuracy of {class_name:<15} : {100 * acc:.2f}%")
                    print("-" * 40)
                
                # 2. ROC 曲线
                if self.enable_roc_curve:
                    all_scores = torch.cat(self.y_scores).numpy()
                    fig_roc = plot_roc_curve(all_trues, all_scores, self.classes)
                    if self.writer is not None:
                        self.writer.add_figure("Eval/ROC_Curve", fig_roc, self.global_step)
                    else:
                        save_path = f"roc_curve_epoch_{self.display_epoch}.png"
                        fig_roc.savefig(save_path)
                        print(f"ROC curve saved to {save_path}")
                    plt.close(fig_roc)

                # 3. 打印 Top-k Accuracy
                if self.top_k is not None:
                    top_k_acc = self.correct_top_k_predictions / self.total_predictions
                    print(f"Top-{self.top_k} Accuracy: {100 * top_k_acc:.2f}%")

                # 清空列表以备下一次评估
                self.y_preds = []
                self.y_trues = []
                self.y_scores = []

        return loss.item()

    def record_history(self, current_val_loss: float = None, current_val_acc: float = None):
        """
        手动将验证集指标添加到 history 字典中。
        """
        if current_val_loss is not None:
            self.history['val_loss'].append(current_val_loss)
        if current_val_acc is not None:
            self.history['val_acc'].append(current_val_acc)

    def predict(self, data_loader: DataLoader, return_targets: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        执行推理并返回所有样本的预测结果。
        
        Args:
            data_loader: 推理数据加载器。
            return_targets: 是否同时也返回标签 (用于计算混淆矩阵等)。
            
        Returns:
            predictions (Tensor): 拼接后的预测结果 (CPU Tensor)。
            targets (Tensor, optional): 拼接后的标签 (CPU Tensor)。
        """
        self.model.eval()
        # 如果使用了 EMA，建议在推理时使用 EMA 的权重 (可选，这里暂不强制覆盖，以免影响后续训练)
        # 你可以手动调用 trainer.ema.apply_shadow(trainer.model) 来应用
        
        preds_list = []
        targets_list = []
        
        print(f"Predicting on {len(data_loader.dataset)} samples...")
        try:
            with torch.no_grad():
                for batch_data in tqdm(data_loader, desc="Predicting", leave=False):
                    self._process_batch_data(batch_data)
                    
                    # Forward
                    with autocast(device_type=self.device.type, enabled=self.use_amp):
                        if isinstance(self.data, tuple):
                            logits = self.model(*self.data)
                        else:
                            logits = self.model(self.data)
                    
                    # 移动到 CPU 以防显存溢出
                    preds_list.append(logits.detach().cpu())
                    
                    if return_targets and self.target is not None:
                        targets_list.append(self.target.detach().cpu())
                        
        finally:
            self.model.train() # 恢复训练模式

        if len(preds_list) == 0:
            return torch.tensor([])

        predictions = torch.cat(preds_list, dim=0)
        
        if return_targets and len(targets_list) > 0:
            targets = torch.cat(targets_list, dim=0)
            return predictions, targets
            
        return predictions

    def auto_checkpoint(self, metrics: Optional[Dict[str, float]] = None, save_best_only: bool = False, monitor: str = 'val_acc') -> None:
        """
        自动保存检查点。
        
        在 Epoch 结束时调用。会保存 'last.pt'。
        如果提供了 metrics 且 monitor 指标优于历史最佳，则保存 'best_model.pt'。

        Args:
            metrics (Dict): 当前 Epoch 的评估指标字典。
            save_best_only (bool): 这里的逻辑通常是只保留 best，但本函数会同时保留 last。
            monitor (str): 监控的指标 key，用于判断最佳模型。
        """
        if not self.is_last_batch_in_epoch or not self.checkpoint_path:
            return
        
        # 1. 保存当前最新状态
        self.save_checkpoint(extra_info=metrics) # 默认保存到 self.checkpoint_path
        
        # 2. 判断是否为最佳模型
        is_best = False
        if metrics and monitor in metrics:
            current_val = metrics[monitor]
            
            # 初始化 best metric
            if self.best_val_metric == -float('inf') and 'loss' in monitor:
                 self.best_val_metric = float('inf')

            if 'loss' in monitor:
                if current_val < self.best_val_metric:
                    self.best_val_metric = current_val
                    is_best = True
            else:
                if current_val > self.best_val_metric:
                    self.best_val_metric = current_val
                    is_best = True
        
        if is_best:
            best_path = os.path.join(os.path.dirname(self.checkpoint_path), 'best_model.pt')
            self.save_checkpoint(path=best_path, extra_info=metrics)
            print(f" -> New best model saved at epoch {self.display_epoch} ({monitor}: {metrics[monitor]:.4f})")
    
    def fit(
            self,
            train_loader: Optional[DataLoader] = None,
            test_loader: Optional[DataLoader] = None,
            cal_classification_metrics: bool = False,
        ) -> 'Trainer':
        """
        傻瓜式训练器。
        支持自动更新、自动保存检查点、自动计算分类指标。
        
        Args:
            train_loader (Optional[DataLoader]): 训练数据加载器。
            test_loader (Optional[DataLoader]): 测试数据加载器。
            cal_classification_metrics (bool): 是否计算分类指标。
            
        Returns:
            self (Trainer): 训练器对象，用于链式调用。
        """
        if not self.model:
            raise RuntimeError("Model missing.")
        if not self.optimizer or not self.criterion:
            raise RuntimeError("Optimizer or Criterion missing.")
        for trainer in self.train(train_loader, tqdm_bar=True, print_loss=True):
            trainer.auto_update()
            trainer.auto_checkpoint()
        
        if cal_classification_metrics:
            for trainer in self.eval(test_loader, tqdm_bar=True):
                trainer.calculate_classification_metrics()
            print(f'Mean Accuracy: {100 * self.eval_accuracy:.2f}%')
        
        return self

    def save_checkpoint(self, path: Optional[str] = None, extra_info: Optional[Dict[str, Any]] = None) -> None:
        path_to_use = path if path is not None else self.checkpoint_path
        if path_to_use is None: return
        
        os.makedirs(os.path.dirname(path_to_use), exist_ok=True)

        # 获取原始模型 state_dict
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_state_dict': self.ema.shadow.state_dict() if (self.use_ema and self.ema) else None,
            
            'history': self.history,
            'best_val_metric': self.best_val_metric,
            'patience_counter': self.patience_counter,
            'best_metric_for_es': self.best_metric_for_es
        }
        if extra_info:
            state.update(extra_info)
        
        try:
            torch.save(state, path_to_use)
        except Exception as e:
            print(f"Error saving checkpoint {path_to_use}: {e}")

    def load_checkpoint(self, path: Optional[str] = None) -> 'Trainer':
        path_to_use = path if path is not None else self.checkpoint_path
        if path_to_use is None or not os.path.exists(path_to_use):
            return self

        print(f"Loading checkpoint: {path_to_use}")
        try:
            checkpoint = torch.load(path_to_use, map_location=self.device)
            
            # 加载权重
            if isinstance(self.model, nn.DataParallel):
                # 如果当前是多卡，加载到 model.module
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 如果当前是单卡，直接加载
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if self.optimizer and checkpoint.get('optimizer_state_dict'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            if self.use_ema and self.ema and checkpoint.get('ema_state_dict'):
                self.ema.shadow.load_state_dict(checkpoint['ema_state_dict'])
                print("EMA state loaded.")

            self.start_epoch = checkpoint.get('epoch', -1) + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.history = checkpoint.get('history', self.history)
            self.best_val_metric = checkpoint.get('best_val_metric', -float('inf'))
            self.patience_counter = checkpoint.get('patience_counter', 0)
            self.best_metric_for_es = checkpoint.get('best_metric_for_es', None)
            
            print(f"Resumed from Epoch {self.display_epoch - 1} (Global Step: {self.global_step}).")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")
            self.start_epoch = 0
        
        return self
    
    def save_model(self, path: str) -> None:
        """
        仅保存模型的权重参数 (state_dict)，用于推理部署。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if isinstance(self.model, nn.DataParallel):
                torch.save(self.model.module.state_dict(), path)
            else:
                torch.save(self.model.state_dict(), path)
            print(f"Model weights saved to: {path}")
        except Exception as e:
            print(f"Error saving model weights to {path}: {e}")

    def load_model(self, path: str, strict: bool = True) -> None:
        """
        加载模型权重。自动处理“纯权重文件”和“完整检查点文件”。
        """
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path}")
            return

        print(f"Loading model weights from: {path}")
        try:
            state_dict = torch.load(path, map_location=self.device)
            
            # 兼容完整 checkpoint 文件
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                print("Detected full checkpoint, extracting 'model_state_dict'...")
                state_dict = state_dict['model_state_dict']
            
            # 兼容处理 DataParallel 加载
            if isinstance(self.model, nn.DataParallel):
                missing, unexpected = self.model.module.load_state_dict(state_dict, strict=strict)
            else:
                missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)
            
            if not strict:
                if missing: print(f"Missing keys (ignored): {len(missing)}")
                if unexpected: print(f"Unexpected keys (ignored): {len(unexpected)}")
            
            print("Model weights loaded successfully.")
            
        except Exception as e:
            print(f"Failed to load model weights: {e}")
