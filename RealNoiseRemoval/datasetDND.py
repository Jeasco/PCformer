import os
import cv2
import torch
import random
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from data_util import RandomCrop, RandomRotation, RandomResizedCrop, RandomHorizontallyFlip, RandomVerticallyFlip
import torchvision.transforms as tfs


class TestDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.dataset = os.path.join(opt.test_dataset, 'test.txt')
        self.image_path = opt.dataset_path

        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)

        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        img_file = file_name.split(' ')[0].strip()

        in_img = cv2.imread(self.image_path + img_file)

        in_img = np.transpose(in_img, (2, 0, 1)).astype(np.float32) / 255.0

        img_path = img_file.split('/')
        sample = {'in_img': in_img, 'image_name':img_path[-1][:-4]}

        return sample

