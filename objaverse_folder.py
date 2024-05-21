from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from typing import Callable, Optional

class ObjaverseDataset(Dataset):
    def __init__(self, root:str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        
        folder_path = self.root
        if not os.path.exists(folder_path):
            print(f"{folder_path} does not exist!!")
            exit()
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            label = image_name.split('.')[0]
            self.images.append(image_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label