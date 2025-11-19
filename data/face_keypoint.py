import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
import json

import os

def label_to_tensor(label):
    """Convert label dictionary to tensor"""
    keypoints = label.get('base_info').get('points_array')
    return torch.tensor(keypoints, dtype=torch.float32).view(106 * 2)

class FaceKeypointsDataset(Dataset):
    def __init__(self, img_dir: str, json_dir: str, transform=None, target_transform=None) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.img_files = []
        self.json_files = []

        for img in os.listdir(img_dir):
            if not img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')): continue
            
            file_name = os.path.splitext(img)[0]
            json_file = os.path.join(json_dir, file_name + '.json')
            
            if os.path.exists(json_file):
                self.img_files.append(os.path.join(img_dir, img))
                self.json_files.append(json_file)
        
        self.num_samples = len(self.img_files)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        json_path = self.json_files[idx]
        
        image = decode_image(img_path).to(dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
            
        with open(json_path, 'r') as f:
            keypoints = json.load(f)
        
        label = keypoints.get('face_infos', [{}])[0] 
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label