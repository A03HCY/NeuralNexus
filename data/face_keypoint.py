# data/face_keypoint.py (完整替换)

import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import numpy as np
from torchvision import transforms

def label_to_tensor(keypoints_array):
    """
    将 NumPy 关键点数组转换为扁平化的 PyTorch Tensor。
    注意：归一化逻辑已移至 Dataset 类中。
    """
    return torch.tensor(keypoints_array, dtype=torch.float32).flatten()

class FaceKeypointsDataset(Dataset):
    def __init__(self, img_dir: str, json_dir: str, transform=None, target_transform=None, img_size=256) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
        
        # --- 新增：定义内部图像变换 ---
        self.resize_transform = transforms.Compose([
            transforms.ToPILImage(), # 先转为 PIL Image 以便 Resize
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()    # 再转回 Tensor [0, 1]
        ])

        self.img_files = []
        self.json_files = []

        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                continue
            
            file_name = os.path.splitext(img_file)[0]
            json_file = os.path.join(json_dir, file_name + '.json')
            
            if os.path.exists(json_file):
                self.img_files.append(os.path.join(img_dir, img_file))
                self.json_files.append(json_file)
        
        self.num_samples = len(self.img_files)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        json_path = self.json_files[idx]
        
        # 使用 PIL 读取图片，更稳定
        image = Image.open(img_path).convert('RGB')
        original_w, original_h = image.size

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        keypoints = np.array(data['face_infos'][0]['base_info']['points_array'], dtype=np.float32)

        # --- 核心修改：统一的 Resize 和归一化流程 ---
        # 1. 对关键点坐标进行缩放以匹配 Resize 后的图像
        keypoints[:, 0] = keypoints[:, 0] * (self.img_size / original_w)
        keypoints[:, 1] = keypoints[:, 1] * (self.img_size / original_h)

        # 2. 对图像进行 Resize
        image_tensor = self.resize_transform(np.array(image)) # (C, H, W)

        # 3. 将调整后的像素坐标归一化到 [0, 1]
        keypoints_normalized = keypoints / self.img_size
        
        # 4. 应用外部传入的 transform (通常是 Normalize)
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        # 5. 应用外部传入的 target_transform (通常是 label_to_tensor)
        if self.target_transform:
            keypoints_final = self.target_transform(keypoints_normalized)
        else:
            keypoints_final = torch.tensor(keypoints_normalized, dtype=torch.float32).flatten()
            
        return image_tensor, keypoints_final
    
    def inverse_transform_keypoints(self, keypoints_tensor):
        """将归一化的 Tensor 关键点转换回像素坐标的 NumPy 数组"""
        if isinstance(keypoints_tensor, torch.Tensor):
            keypoints_tensor = keypoints_tensor.detach().cpu()
        
        keypoints_np = keypoints_tensor.view(-1, 2).numpy()
        keypoints_np = keypoints_np * self.img_size
        return keypoints_np

