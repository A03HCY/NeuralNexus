import copy
import torch
import torch.nn as nn

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
