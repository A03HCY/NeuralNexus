import json
import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class AnswersDataset(Dataset):
    """
    一个用于处理特定格式JSONL文件的PyTorch Dataset。
    
    JSONL文件每行包含一个JSON对象，格式为：
    {"question": "...", "human_answers": ["..."], "chatgpt_answers": ["..."]}
    
    该Dataset将忽略"question"，并将"human_answers"和"chatgpt_answers"
    中的每个回答作为一个独立的样本。
    
    标签 (One-hot encoded Tensor):
    - torch.tensor([1, 0]): 人类回答 (human_answers)
    - torch.tensor([0, 1]): AI回答 (chatgpt_answers)
    """
    def __init__(self, jsonl_path):
        """
        Args:
            jsonl_path (str): JSONL文件的路径。
        """
        self.samples: List[Tuple[str, int]]= []
        # 定义内部使用的整数标签
        # 1 代表 人类, 0 代表 AI
        self.human_label_int = 1
        self.ai_label_int = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                # 处理人类回答
                for answer in data['human_answers']:
                    self.samples.append((answer.lower(), self.human_label_int))
                
                # 处理ChatGPT回答
                for answer in data['chatgpt_answers']:
                    self.samples.append((answer.lower(), self.ai_label_int))
    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.samples)
    def __getitem__(self, idx):
        """
        根据索引idx获取一个样本，并将标签转换为one-hot tensor。
        
        Args:
            idx (int): 样本的索引。
        
        Returns:
            tuple: (回答文本, 标签Tensor)
        """
        text, label_int = self.samples[idx]
        
        # 将整数标签转换为 one-hot encoded tensor
        if label_int == self.human_label_int:
            # 人类标签: [1, 0]
            label_tensor = torch.tensor([1, 0], dtype=torch.float32)
        else:
            # AI标签: [0, 1]
            label_tensor = torch.tensor([0, 1], dtype=torch.float32)
            
        return text, label_tensor