import torch
import random
import numpy as np

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
