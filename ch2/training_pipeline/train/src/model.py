import logging
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Cifar10Dataset(Dataset):
    def __init__(self, data_dir, transform):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self.__load_data()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]

    def __load_data(self):
        class_dirs = [i for i in os.listdir(self.data_dir) if i.isdecimal()]
        file_paths = []
        
        for d in class_dirs:
            _d = os.path.join(self.data_dir, d)
            file_paths.extend([os.path.join(_d, f) for f in os.listdir(_d)])
            self.labels.extend([int(d) for _ in os.listdir(_d)])
        
        for fp in file_paths:
            with Image.open(fp, "r") as img:
                self.images.append(np.array(img))
        
        logger.info(f"loaded: {len(self.labels)} data")

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.con1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x