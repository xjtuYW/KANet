import os
import PIL.Image as Image
import numpy as np
import torch

__all__ = ['CategoryDatasetFolder']


class CategoryDatasetFolder(object):

    def __init__(self, 
                 data_root, 
                 file_name, 
                 transform, 
                 dataset_min_cls: int=0, 
                 dataset_max_cls: int=10000,
                 out_name: bool=False):
        data, ori_labels = [], []
        file_name        = file_name + '.csv'
        file_csv         = os.path.join(data_root, file_name)
        with open(file_csv, 'r') as f:  # (name.jpg,label)
            next(f) # skip first line
            for split in f.readlines():
                split = split.strip('\n').split(',')  # split = [name.jpg label]
                if int(split[1]) >= dataset_min_cls and int(split[1]) < dataset_max_cls:
                    data.append(os.path.join(data_root, split[0]))  # name.jpg
                    ori_labels.append(int(split[1]))  # label
                    
        self.file_name  = file_name
        self.data_root  = data_root
        self.transform  = transform
        self.data       = data
        self.labels     = ori_labels
        self.out_name   = out_name
        self.length     = len(self.data)

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        img_path = self.data[index]

        img = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            return img, label, self.data[index]
        else:
            return img, label

