import torch
import torch.nn as nn
from torch.nn.modules import conv
from models.submodules.net_basics import *

class Event_encoder_block(nn.Module):
    def __init__(self, num_bins, nf):
        super(Event_encoder_block, self).__init__()
        # hidden dims
        hidden_dims = nf
        # convolution block
        self.conv0 = conv3x3_leaky_relu(num_bins, nf)
        self.conv1 = conv_resblock_two(nf, nf, stride=1)
        self.conv2 = conv_resblock_two(nf, 2*nf, stride=2) 
        self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)
        self.conv4 = conv_resblock_two(4*nf+hidden_dims, 8*nf, stride=2)
        # hidden state
        self.hidden_conv = conv_resblock_two(4*nf+nf, nf, stride=1)

    def forward(self, x, x_hidden):
        x_ = self.conv0(x)
        # feature computation
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        x_in = torch.cat((f3, x_hidden), dim=1)
        f4 = self.conv4(x_in)
        # hidden state computation
        x_hidden_out = self.hidden_conv(x_in)
        return [f1, f2, f3, f4], x_hidden_out

class Past_Event_encoder_block(nn.Module):
    def __init__(self, num_bins, nf):
        super(Past_Event_encoder_block, self).__init__()
        # convolution block
        self.conv0 = conv3x3_leaky_relu(num_bins, nf)
        self.conv1 = conv_resblock_two(nf, nf, stride=1)
        self.conv2 = conv_resblock_two(nf, 2*nf, stride=2)
        self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)
        self.conv4 = conv_resblock_two(4*nf, 8*nf, stride=2)

    def forward(self, x):
        x_ = self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        return [f1, f2, f3, f4]
        
class Event_encoder(nn.Module):
    def __init__(self, voxel_num_bins, nf):
        super(Event_encoder, self).__init__()
        self.Unit_event_encoder = Event_encoder_block(2, nf) # 2, 8
        self.Past_event_encoder = Past_Event_encoder_block(nf*2, nf)# 16,8
        self.voxel_num_bins = voxel_num_bins//2
        # other parameter
        self.ds_ratio = 4
        self.nf = nf
    
    def forward(self, x_event):
        B, _, H, W = x_event.shape
        f_e = []
        x_event_past = x_event[:, :self.voxel_num_bins, ...] # t0   16
        x_event_cur = x_event[:, self.voxel_num_bins:, ...] # t1    16
        f_past = self.Past_event_encoder(x_event_past)
        f_e.append(f_past)
        # hidden state
        f_hidden = torch.zeros(B, self.nf, H//self.ds_ratio, W//self.ds_ratio).cuda()
        for i in range(8):
            x_event_tmp = x_event_cur[:, 2*i:2*(i+1), ...]# B, t_i, ...
            f_cur, f_hidden = self.Unit_event_encoder(x_event_tmp, f_hidden)
            f_e.append(f_cur)
        return f_e


class Blur_encoder_v2(nn.Module):
    def __init__(self, in_dims, nf):
        super(Blur_encoder_v2, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_two(nf, nf, stride=1)
        self.conv2 = conv_resblock_two(nf, 2*nf, stride=2)
        self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)
        self.conv4 = conv_resblock_two(4*nf, 8*nf, stride=2)
    
    def forward(self, x):
        x_ = self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        return [f1, f2, f3, f4]


class Decoder_v2(nn.Module):
    def __init__(self, nf):
        super(Decoder_v2, self).__init__()
        self.deconv_1_1 = deconv4x4(16*nf, 8*nf)
        self.deconv_1_2 = conv_resblock_two(16*nf, 8*nf, stride=1)

        self.deconv_2_1 = deconv4x4(8*nf, 4*nf)
        self.deconv_2_2 = conv_resblock_two(8*nf, 4*nf)

        self.deconv_3_1 = deconv4x4(4*nf, 2*nf)
        self.deconv_3_2 = conv_resblock_two(4*nf, 2*nf)

        self.deconv_4_1 = conv_resblock_two(2*nf, nf)

        self.nf = nf

    def forward(self, x_input):
        x_ = self.deconv_1_1(x_input[3])
        x_in = torch.cat((x_, x_input[2]), dim=1)
        x_ = self.deconv_1_2(x_in)

        x_ = self.deconv_2_1(x_)
        x_in = torch.cat((x_, x_input[1]), dim=1)
        x_ = self.deconv_2_2(x_in)

        x_ = self.deconv_3_1(x_)
        x_in = torch.cat((x_, x_input[0]), dim=1)
        x_ = self.deconv_3_2(x_in)

        x_out = self.deconv_4_1(x_)
        return x_out