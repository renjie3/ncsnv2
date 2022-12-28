import torch
import os
from PIL import Image
from torchvision.datasets import CIFAR10
import numpy as np

class AdvCifar10(CIFAR10):
    def __init__(self, root, train=True, download=True, transform=None, args=None, config=None):
        super().__init__(root, train=train, download=download, transform=transform)
        self.transform = transform
        self.config = config
        self.args = args
        self.train = train

        if config.data.sub_dataset and self.train:
            if config.data.subset_number == 0:
                sub_class = [2, 7, 9]
                single_class_num = 3000
                np_targets = np.array(self.targets)
                new_data_list = []
                new_targets_list = []
                for i in range(len(sub_class)):
                    sub_class_idx = np_targets == sub_class[i]
                    new_data_list.append(self.data[sub_class_idx][:single_class_num])
                    new_targets_list.append(np_targets[sub_class_idx][:single_class_num])
                self.data = np.concatenate(new_data_list, axis=0)
                self.targets = np.concatenate(new_targets_list, axis=0)
                
                for i in range(len(sub_class)):
                    sub_class_idx = self.targets == sub_class[i]
                    self.targets[sub_class_idx] = i

            else:
                raise("unknown subset_number")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx] # 0 to 255
        img = Image.fromarray(img)

        if self.transform != None:
            img = self.transform(img)

        label = np.array(self.targets[idx], dtype=np.int64)
        return img, label, idx
