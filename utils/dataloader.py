import os 
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import torch
import random
import time
from PIL import Image
from utils.utils import *


class DataLoader_dvs_train(data.Dataset):
    def __init__(self, data_dir, mode, filename, num_bins, training):
        # blur image sequence
        self.blur_images_0 = [os.path.join(data_dir, mode, line.split(' ')[0], 'blur_images', line.split(' ')[1],  \
                              line.split(' ')[2], line.split(' ')[3]) for line in open(filename)]
        self.blur_images_1 = [os.path.join(data_dir, mode, line.split(' ')[0], 'blur_images', line.split(' ')[1],  \
                              line.split(' ')[2], line.split(' ')[4].split('\n')[0]) for line in open(filename)]
        # event data
        self.events_vox_0 = [os.path.join(data_dir, mode, line.split(' ')[0], 'event_voxel', \
                            line.split(' ')[1], line.split(' ')[2], line.split(' ')[3]) for line in open(filename)]
        self.events_vox_1 = [os.path.join(data_dir, mode, line.split(' ')[0], 'event_voxel',  \
                            line.split(' ')[1], line.split(' ')[2], line.split(' ')[4].split('\n')[0]) for line in open(filename)]
        # clean gt images
        self.clean_images_gt = [os.path.join(data_dir, mode, line.split(' ')[0], 'clean_gt_images', line.split(' ')[1], \
                                line.split(' ')[2], line.split(' ')[4].split('\n')[0]) for line in open(filename)]
        # transform
        self.transform = transforms.ToTensor()
        self.training = training
        self.num_bins = num_bins
        # data augmenation parameter
        self.crop_width = 256
        self.crop_height = 256
        # time interval list
        self.time_interval = [line.split(' ')[2] for line in open(filename)]

    def __getitem__(self, idx):
        sample = {}
        # blur image
        B0 = Image.open(self.blur_images_0[idx].zfill(5)+'.png')
        B1 = Image.open(self.blur_images_1[idx].zfill(5)+'.png')
        # transform blur image
        B0 = self.transform(B0)
        B1 = self.transform(B1)
        B_all = torch.cat((B0, B1), dim=0)
        # Image open
        C_gt = Image.open(self.clean_images_gt[idx].zfill(5) + '.png')
        width, height = C_gt.size
        C_gt = self.transform(C_gt)
        # event
        EV_tensor_0 = torch.from_numpy(np.load(self.events_vox_0[idx].zfill(5) + '.npz')["data"])
        EV_tensor_1 = torch.from_numpy(np.load(self.events_vox_1[idx].zfill(5) + '.npz')["data"])
        EV_tensor = torch.cat((EV_tensor_0, EV_tensor_1), dim=0)
        if self.training:
            # random cropping..
            x = random.randint(0, width - self.crop_width)
            y = random.randint(0, height - self.crop_height)
            B_all = randomCrop(B_all, x, y, self.crop_width, self.crop_height)
            C_gt = randomCrop(C_gt, x, y, self.crop_width, self.crop_height)
            EV_tensor = randomCrop(EV_tensor, x, y, self.crop_width, self.crop_height)
            sample['blur_image'] = B_all
            sample['voxel_grid'] = EV_tensor
            sample['clean_gt_images'] = C_gt
            time_interval = self.time_interval[idx]
            return sample, time_interval
        else:
            sample['blur_image'] = B_all
            sample['voxel_grid'] = EV_tensor
            sample['clean_gt_images'] = C_gt
            time_interval = self.time_interval[idx]
            return sample, time_interval

    def __len__(self):
        return len(self.blur_images_0)


class DataLoader_dvs_test(data.Dataset):
    def __init__(self, data_dir, mode, filename, num_bins, training):
        # blur image sequence
        self.blur_images_0 = [os.path.join(data_dir, mode, line.split(' ')[0], 'blur_images', line.split(' ')[1], \
                              line.split(' ')[2], line.split(' ')[3] + '.png') for line in open(filename)]
        self.blur_images_1 = [os.path.join(data_dir, mode, line.split(' ')[0], 'blur_images', line.split(' ')[1], \
                              line.split(' ')[2], line.split(' ')[4].split('\n')[0] + '.png') for line in open(filename)]
        # event data
        self.events_vox_0 = [os.path.join(data_dir, mode, line.split(' ')[0], 'event_voxel', \
                            line.split(' ')[1], line.split(' ')[2], line.split(' ')[3] + '.npz') for line in open(filename)]
        self.events_vox_1 = [os.path.join(data_dir, mode, line.split(' ')[0], 'event_voxel',  \
                            line.split(' ')[1], line.split(' ')[2], line.split(' ')[4].split('\n')[0] + '.npz') for line in open(filename)]
        # clean gt images
        self.clean_images_gt = [os.path.join(data_dir, mode, line.split(' ')[0], 'clean_gt_images', line.split(' ')[1], \
                                line.split(' ')[2], line.split(' ')[4].split('\n')[0] + '.png') for line in open(filename)]
        # transform
        self.transform = transforms.ToTensor()
        self.training = training
        self.num_bins = num_bins
        # time interval list
        self.time_interval = [line.split(' ')[2] for line in open(filename)]

    def __getitem__(self, idx):
        sample = {}
        # blur image
        B0 = Image.open(self.blur_images_0[idx])
        B1 = Image.open(self.blur_images_1[idx])
        # transform blur image
        B0 = self.transform(B0)
        B1 = self.transform(B1)
        B_all = torch.cat((B0, B1), dim=0)
        # Image open
        C_gt = Image.open(self.clean_images_gt[idx])
        # transform
        C_gt = self.transform(C_gt)
        # total event
        EV_tensor_0 = torch.from_numpy(np.load(self.events_vox_0[idx])["data"])
        EV_tensor_1 = torch.from_numpy(np.load(self.events_vox_1[idx])["data"])
        EV_tensor = torch.cat((EV_tensor_0, EV_tensor_1), dim=0)
        sample['blur_image'] = B_all
        sample['voxel_grid'] = EV_tensor
        sample['clean_gt_images'] = C_gt
        time_interval = self.time_interval[idx]
        return sample, time_interval

    def __len__(self):
        return len(self.blur_images_0)