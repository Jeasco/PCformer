import os
import cv2
import torch
import random
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from data_util import RandomCrop, RandomCropOne,RandomRotation, RandomResizedCrop, RandomHorizontallyFlip, RandomVerticallyFlip


class TrainDataset(Dataset):
    def __init__(self, opt):
        super().__init__()

        self.dataset = os.path.join(opt.train_dataset, 'imagenet.txt')
        self.image_path = opt.train_dataset
        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.image_size = opt.image_size

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_1 = self.mat_files[idx % self.file_num]
        file_2 = self.mat_files[random.randint(0, self.file_num)]

        img_1 = cv2.imread(self.image_path + file_1.strip())
        img_2 = cv2.imread(self.image_path + file_2.strip())

        ps = self.image_size // 2

        img_1 = Image.fromarray(img_1)
        w, h = img_1.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            img_1 = TF.pad(img_1, (0, 0, padw, padh), padding_mode='reflect')


        img_1 = TF.to_tensor(img_1)

        hh, ww = img_1.shape[1], img_1.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)

        # Crop patch
        img_1_1 = img_1[:, rr:rr + ps, cc:cc + ps]
        tar_1_1 = img_1_1.clone()

        img_2 = Image.fromarray(img_2)
        w, h = img_2.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            img_2 = TF.pad(img_2, (0, 0, padw, padh), padding_mode='reflect')

        img_2 = TF.to_tensor(img_2)

        hh, ww = img_2.shape[1], img_2.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)

        # Crop patch
        img_2_1 = img_2[:, rr:rr + ps, cc:cc + ps]
        tar_2_1 = img_2_1.clone()

        for _ in range(3):
            w = random.randint(5, 32)
            h = random.randint(5, 32)
            rr = random.randint(0, ps - w)
            cc = random.randint(0, ps - h)
            img_1_1[:, rr:rr + w, cc:cc + h] = torch.rand(3,w,h)
            w = random.randint(5, 32)
            h = random.randint(5, 32)
            rr = random.randint(0, ps - w)
            cc = random.randint(0, ps - h)
            img_2_1[:, rr:rr + w, cc:cc + h] = torch.rand(3, w, h)

        img_1_2 = img_1_1.clone()
        img_2_2 = img_2_1.clone()
        tar_1_2 = tar_1_1.clone()
        tar_2_2 = tar_2_1.clone()

        for _ in range(3):
            w = random.randint(5, 32)
            h = random.randint(5, 32)
            rr = random.randint(0, ps - w)
            cc = random.randint(0, ps - h)
            img_1_1[:, rr:rr + w, cc:cc + h] = torch.rand(3, w, h)
            w = random.randint(5, 32)
            h = random.randint(5, 32)
            rr = random.randint(0, ps - w)
            cc = random.randint(0, ps - h)
            img_2_1[:, rr:rr + w, cc:cc + h] = torch.rand(3, w, h)
            w = random.randint(5, 32)
            h = random.randint(5, 32)
            rr = random.randint(0, ps - w)
            cc = random.randint(0, ps - h)
            img_1_2[:, rr:rr + w, cc:cc + h] = torch.rand(3, w, h)
            w = random.randint(5, 32)
            h = random.randint(5, 32)
            rr = random.randint(0, ps - w)
            cc = random.randint(0, ps - h)
            img_2_2[:, rr:rr + w, cc:cc + h] = torch.rand(3, w, h)

        img_1_1, tar_1_1 = aug_data(img_1_1, tar_1_1)
        img_1_2, tar_1_2 = aug_data(img_1_2, tar_1_2)
        img_2_1, tar_2_1 = aug_data(img_2_1, tar_2_1)
        img_2_2, tar_2_2 = aug_data(img_2_2, tar_2_2)

        inp_img = torch.zeros(3,self.image_size,self.image_size)
        tar_img = torch.zeros(3,self.image_size,self.image_size)
        mode = random.randint(0, 3)
        if mode == 0:
            inp_img[:, 0:ps, 0:ps] = img_1_1
            inp_img[:, 0:ps, ps:] = img_1_2
            inp_img[:, ps:, 0:ps] = img_2_1
            inp_img[:, ps:, ps:] = img_2_2
            tar_img[:, 0:ps, 0:ps] = tar_1_1
            tar_img[:, 0:ps, ps:] = tar_1_2
            tar_img[:, ps:, 0:ps] = tar_2_1
            tar_img[:, ps:, ps:] = tar_2_2
        elif mode == 1:
            inp_img[:, 0:ps, 0:ps] = img_1_1
            inp_img[:, 0:ps, ps:] = img_2_1
            inp_img[:, ps:, 0:ps] = img_1_2
            inp_img[:, ps:, ps:] = img_2_2
            tar_img[:, 0:ps, 0:ps] = tar_1_1
            tar_img[:, 0:ps, ps:] = tar_2_1
            tar_img[:, ps:, 0:ps] = tar_1_2
            tar_img[:, ps:, ps:] = tar_2_2
        elif mode == 2:
            inp_img[:, 0:ps, 0:ps] = img_1_1
            inp_img[:, 0:ps, ps:] = img_2_1
            inp_img[:, ps:, 0:ps] = img_2_2
            inp_img[:, ps:, ps:] = img_1_2
            tar_img[:, 0:ps, 0:ps] = tar_1_1
            tar_img[:, 0:ps, ps:] = tar_2_1
            tar_img[:, ps:, 0:ps] = tar_2_2
            tar_img[:, ps:, ps:] = tar_1_2

        sample = {'in_img': inp_img, 'gt_img': tar_img}

        return sample

def aug_data(inp_img, tar_img):
    aug = random.randint(0, 8)
    # Data Augmentations
    if aug == 1:
        inp_img = inp_img.flip(1)
        tar_img = tar_img.flip(1)
    elif aug == 2:
        inp_img = inp_img.flip(2)
        tar_img = tar_img.flip(2)
    elif aug == 3:
        inp_img = torch.rot90(inp_img, dims=(1, 2))
        tar_img = torch.rot90(tar_img, dims=(1, 2))
    elif aug == 4:
        inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
        tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
    elif aug == 5:
        inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
        tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
    elif aug == 6:
        inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
        tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
    elif aug == 7:
        inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
        tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
    return inp_img, tar_img