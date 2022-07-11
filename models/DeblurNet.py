from utils.utils import AverageMeter
import torch.nn as nn
import torch
import os
from models.submodules.submodule import *
from math import ceil
import importlib

class UnknwonDeblurNet(object):
    def __init__(self, voxel_num_bins, flag_ES):
        super(UnknwonDeblurNet, self).__init__()
        # intitilize batch input
        self.batch = {}
        # flag ES
        self.flag_ES = flag_ES
        # hyper-parameter
        self.num_encoded_frame = 2
        self.voxel_num_bins = voxel_num_bins
        # loss meter
        self.loss_total_meter = AverageMeter()
        # downsample op
        self.downsample = nn.AvgPool2d(2, stride=2)
        # scale
        self.scale = 3 
        # lamba for scale
        self.loss_weight = [1, 0.1, 0.1]

    def initalize(self, model_folder, model_name, tb_path):
        # load model
        mod = importlib.import_module('models.' + model_folder + '.' + model_name)
        self.model = mod.EvOurUnknownDeblurNet(self.voxel_num_bins, self.flag_ES)
        # save path
        if tb_path is not None:
            self.save_path = os.path.join(tb_path, 'saved_model')
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
    
    def cuda(self): # set cuda device
        self.model.cuda()
    
    def train(self): # train mode
        self.model.train()
    
    def eval(self): # evaluation mode
        self.model.eval()
    
    def get_optimizer_params(self): # get optimizer parameter
        return self.model.parameters()
    
    def use_multi_gpu(self): # data parallel
        self.model = nn.DataParallel(self.model)
    
    def set_input(self, sample, time_interval):
        # gt clean images
        self.batch['clean_gt_images'] = sample['clean_gt_images']
        # blur images and event voxel grid
        self.batch['blur_image_input'] = sample['blur_image']
        # event voxel grid
        self.batch['event_voxel_grid'] = sample['voxel_grid']
        # generate multi_scale gt
        self.batch['clean_gt_MS_images'] = []
        labels = sample['clean_gt_images']
        self.batch['clean_gt_MS_images'].append(sample['clean_gt_images'])
        # set input
        self.batch['time_interval'] = time_interval
        for _ in range(self.scale-1):
            labels = self.downsample(labels.clone())
            self.batch['clean_gt_MS_images'].append(labels)
    
    def set_test_input_real(self, sample, time_interval):
        # shape configuration
        B, _, H, W = sample['blur_image'].shape
        # gt clean images
        self.batch['clean_gt_images'] = sample['clean_gt_images']
        H_ = ceil(H/16)*16
        W_ = ceil(W/16)*16
        _B0 = torch.zeros((B, 3, H_, W_)).cuda()
        _B1 = torch.zeros((B, 3, H_, W_)).cuda()
        _B0[:, :, 0:H, 0:W] = sample['blur_image'][:, :3, ...]
        _B1[:, :, 0:H, 0:W] = sample['blur_image'][:, 3:, ...]
        B_all = torch.cat((_B0, _B1), dim=1)
        self.batch['blur_image_input'] = B_all
        self.batch['event_voxel_grid'] = torch.zeros((B, self.num_encoded_frame*self.voxel_num_bins, H_, W_)).cuda()
        self.batch['event_voxel_grid'][:, :, 0:H, 0:W] = sample['voxel_grid']
        # parameter configuations
        self.H_org = H
        self.W_org = W
        # set input
        self.batch['time_interval'] = time_interval

    def set_test_input_real_shot(self, sample):
        # shape configuration
        B, _, H, W = sample['blur_image'].shape
        H_ = ceil(H/16)*16
        W_ = ceil(W/16)*16
        _B0 = torch.zeros((B, 3, H_, W_)).cuda()
        _B1 = torch.zeros((B, 3, H_, W_)).cuda()
        _B0[:, :, 0:H, 0:W] = sample['blur_image'][:, :3, ...]
        _B1[:, :, 0:H, 0:W] = sample['blur_image'][:, 3:, ...]
        B_all = torch.cat((_B0, _B1), dim=1)
        self.batch['blur_image_input'] = B_all
        self.batch['event_voxel_grid'] = torch.zeros((B, self.num_encoded_frame*self.voxel_num_bins, H_, W_)).cuda()
        self.batch['event_voxel_grid'][:, :, 0:H, 0:W] = sample['voxel_grid']
        # parameter configuations
        self.H_org = H
        self.W_org = W

    def forward(self):
        if self.flag_ES:
            self.batch['clean_image_est'], self.batch['ca_map_list'] = self.model(self.batch['event_voxel_grid'], self.batch['blur_image_input'])
        else:
            self.batch['clean_image_est'] = self.model(self.batch['event_voxel_grid'], self.batch['blur_image_input'])

    def forward_test_real(self):
        if self.flag_ES:
            self.batch['clean_image_est'], self.batch['ca_map_list'] = self.model(self.batch['event_voxel_grid'], self.batch['blur_image_input'])
        else: 
            self.batch['clean_image_est'] = self.model(self.batch['event_voxel_grid'], self.batch['blur_image_input'])
        self.batch['clean_image_est_'] = self.batch['clean_image_est'][0][..., 0:self.H_org,0:self.W_org]

    def get_chainbor_loss(self, x, y):
        loss = ((((x - y) ** 2 + 1e-6) ** 0.5).mean())
        return loss
    
    def get_single_loss(self):
        self.loss = self.get_chainbor_loss(self.batch['clean_gt_images'], self.batch['clean_image_est'][0])
        return self.loss
    
    def get_multi_scale_loss(self):
        self.loss = 0
        for i in range(self.scale):
            self.loss += self.loss_weight[i]*self.get_chainbor_loss(self.batch['clean_gt_MS_images'][i], self.batch['clean_image_est'][i])
        return self.loss

    def update_loss_meters(self):
        # total loss update
        self.loss_total_meter.update(self.loss.item(), 1)
    
    def reset_loss_meters(self):
        self.loss_total_meter.reset()
    
    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))

    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)
        print('load model')