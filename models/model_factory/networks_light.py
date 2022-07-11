import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules import conv
from models.submodules.net_basics import *
import torch.nn.functional as F
from libs.kernelconv2d import KernelConv2D


class ca_layer_act(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer_act, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y

class Blur_encoder_v3(nn.Module):
    def __init__(self, in_dims, nf):
        super(Blur_encoder_v3, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_one(nf, nf)
        self.conv2 = conv_resblock_one(nf, 2*nf, stride=2)
        self.conv3 = conv_resblock_one(2*nf, 4*nf, stride=2)
    
    def forward(self, x):
        x_ =  self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]

class Past_Event_encoder_block(nn.Module):
    def __init__(self, num_bins, nf):
        super(Past_Event_encoder_block, self).__init__()
        # convolution block
        self.conv0 = conv3x3_leaky_relu(num_bins, nf)
        self.conv1 = conv3x3_leaky_relu(nf, nf, stride=1)
        self.conv2 = conv3x3_leaky_relu(nf, 2*nf, stride=2)
        self.conv3 = conv3x3_leaky_relu(2*nf, 4*nf, stride=2)

    def forward(self, x):
        x_ = self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]

class Event_encoder_block(nn.Module):
    def __init__(self, num_bins, nf):
        super(Event_encoder_block, self).__init__()
        # hidden dims
        self.hidden_dims = nf//2
        # convolution block
        self.conv0 = conv3x3_leaky_relu(num_bins, nf)
        self.conv1 = conv_resblock_one(nf, nf, stride=1)
        self.conv2_0 = nn.Sequential(conv3x3(nf, 2*nf, stride=2), nn.ReLU())
        self.conv2_1 = conv_resblock_one(2*nf+self.hidden_dims, 2*nf)
        self.conv3 = conv_resblock_one(2*nf, 4*nf, stride=2)
        # hidden state
        self.hidden_conv = conv3x3(2*nf, self.hidden_dims, stride=1)

    def forward(self, x, x_hidden):
        x_ = self.conv0(x)
        # feature computation
        f1 = self.conv1(x_)
        f2_0 = self.conv2_0(f1)
        x_in = torch.cat((f2_0, x_hidden), dim=1)
        f2 = self.conv2_1(x_in)
        f3 = self.conv3(f2)
        # hidden state computation
        x_hidden_out = self.hidden_conv(f2)
        return [f1, f2, f3], x_hidden_out

class Event_encoder(nn.Module):
    def __init__(self, voxel_num_bins, nf):
        super(Event_encoder, self).__init__()
        # other parameter
        self.ds_ratio = 2
        # number of iteration 
        self.voxel_num_bins = voxel_num_bins
        # unit encoders
        self.Unit_event_encoder = Event_encoder_block(2, nf)
        # past encoder
        self.Past_event_encoder = Past_Event_encoder_block(voxel_num_bins, nf)
        # number iteration
        self.num_iter = 8
        # number of feature
        self.nf = nf
    
    def forward(self, x_event):
        B, _, H, W = x_event.shape
        f_e = []
        x_event_past = x_event[:, :self.voxel_num_bins, ...] # t0   16
        x_event_cur = x_event[:, self.voxel_num_bins:, ...] # t1    16
        f_past = self.Past_event_encoder(x_event_past)
        f_e.append(f_past)
        # hidden state 
        f_hidden = torch.zeros(B, self.nf//2, H//2, W//2).cuda()
        for i in range(self.num_iter):
            x_event_tmp = x_event_cur[:, 2*i:2*(i+1), ...]
            f_cur, f_hidden = self.Unit_event_encoder(x_event_tmp, f_hidden)
            f_e.append(f_cur)
        return f_e

class Event_selection(nn.Module):
    def __init__(self, scale=3, n_feat=4):
        super(Event_selection, self).__init__()
        # unit queue length 
        self.unit_queue_channel = n_feat
        self.n_feat_total = 9*n_feat
        self.ba_extractor = nn.ModuleList([conv1x1(self.n_feat_total*2**(i),self.unit_queue_channel) for i in range(scale)])
        self.shared_ea_extractor = nn.ModuleList([conv_resblock_one(self.unit_queue_channel*2**(i),self.unit_queue_channel) for i in range(3)]) ## FIX
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # module body for multi-scale processing..
        modules_body_0   = [nn.Conv2d(self.n_feat_total, self.n_feat_total, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(), \
                            nn.Conv2d(self.n_feat_total, self.n_feat_total, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(), \
                            nn.Conv2d(self.n_feat_total, self.n_feat_total, kernel_size=3, stride=1, padding=1, bias=True)]
        modules_body_1   = [nn.Conv2d(self.n_feat_total, self.n_feat_total, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(), \
                            nn.Conv2d(self.n_feat_total, self.n_feat_total, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(), \
                            nn.Conv2d(self.n_feat_total, self.n_feat_total, kernel_size=3, stride=1, padding=1, bias=True)]
        self.conv_ba_extractors_0 = nn.Sequential(*modules_body_0)
        self.conv_ba_extractors_1 = nn.Sequential(*modules_body_1)

        # upsampling function
        self.up_2x = nn.Upsample(scale_factor=2, mode='nearest')

        # self.ca_layer_act = nn.ModuleList([ca_layer_act(8*2**(i)) for i in range(3)])
        self.ea_norm = nn.ModuleList([nn.GroupNorm(9,self.n_feat_total, affine=False) for i in range(3)]) ## FIX
        self.ba_norm = nn.ModuleList([nn.GroupNorm(1,self.unit_queue_channel,affine=False) for i in range(3)]) ## FIX

    def forward(self, f_event, f_blur):
        fb_fe_corr_list = []
        B, T, C, H, W = f_event[0].shape
        y_act_list = []
        for idx, (fe, fb) in enumerate(zip(f_event, f_blur)): # 2~0scale
            # past ~ cur
            fb_contract=self.ba_extractor[idx](fb) # 8,16,32,64 ###### HR -> LR
            fb_contract=self.ba_norm[idx](fb_contract)
            fb_nd=fb_contract.repeat(1,9,1,1)
            f_e = []
            # y_act = []
            for queue_idx in range(9):
                x_event_tmp = fe[:, queue_idx, ...]
                f_cur = self.shared_ea_extractor[idx](x_event_tmp)
                f_e.append(f_cur)
            fe_processed = torch.cat((f_e), dim=1)
            fe_processed=self.ea_norm[idx](fe_processed)
            fb_fe_corr= fe_processed*fb_nd # 72, 144, 256
            fb_fe_corr_list.append(fb_fe_corr)
        s2 = fb_fe_corr_list[2]
        s1 = fb_fe_corr_list[1]
        ## mixing 2->1
        s2_up = self.up_2x(s2)
        s1s2_sum = s1 + s2_up
        s1s2_sum_refined = s1s2_sum + self.conv_ba_extractors_0(s1s2_sum) # skip connection
        s0 = fb_fe_corr_list[0]
        ## mixing 1->0
        fb_fe_corr_sum = s0+self.up_2x(s1s2_sum_refined)
        corr_map = fb_fe_corr_sum + self.conv_ba_extractors_1(fb_fe_corr_sum)
        c_map = self.avg_pool(corr_map)
        ca_map = torch.sigmoid(c_map) 
        ca_map_list, f_event_attended = [], []
        temp_ca_map = ca_map
        for idx, fe in enumerate(f_event): # 2~0scale
            if idx >0:
                temp_ca_map=ca_map.squeeze(2) # B,72,1,1  -> B,72,1
                temp_ca_map=temp_ca_map.permute(0,2,1) # B,1,72
                temp_ca_map=F.interpolate(temp_ca_map,[self.n_feat_total*(2**idx)],mode='nearest') # B,1,144
                temp_ca_map=temp_ca_map.permute(0,2,1) # B,144,1
            temp_ca_map=temp_ca_map.reshape(B,-1).unsqueeze(-1).unsqueeze(-1) # 3,72,2,1 -> 3,144,1,1
            fe = fe.view(fe.shape[0], fe.shape[1]*fe.shape[2], fe.shape[3], fe.shape[4])
            f_event_attended.append(fe*temp_ca_map)# 3
            ca_map_list.append(temp_ca_map)
        return f_event_attended, ca_map_list

class DAU_ours_v5(nn.Module):
    def __init__(self, n_feat, kernel_size=3, filter_ks=3, reduction=8, bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(DAU_ours_v5, self).__init__()
        modules_body_0 = [conv(n_feat, n_feat, kernel_size, bias=bias), nn.ReLU(), conv(n_feat, n_feat, kernel_size, bias=bias)]
        modules_body_1 = [conv(n_feat, n_feat, kernel_size, bias=bias), nn.ReLU()]
        # boda sa
        self.trans = nn.Sequential(*modules_body_0)
        self.trans2 = nn.Sequential(*modules_body_1)
        self.F_mlp = nn.Sequential(nn.Linear(n_feat, 2*n_feat), nn.ReLU(), nn.Linear(2*n_feat, n_feat), nn.Sigmoid())
        ## dynamic filtering
        self.kernel_size = filter_ks
        self.kconv = KernelConv2D.KernelConv2D(kernel_size=self.kernel_size)
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.ReLU(), 
            nn.Conv2d(n_feat, n_feat*self.kernel_size**2, 1, 1)
        )
        ## Spatial Attention
        self.gate_rgb = nn.Conv2d(n_feat, 1, kernel_size=1, bias=True)
        ### self attention 
        modules_body_2 = [conv(n_feat, n_feat, kernel_size, bias=bias), nn.ReLU(), conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body_sa = nn.Sequential(*modules_body_2)
        ## compression
        self.compress = ChannelPool()
        ## event conv
        self.spatial_e = BasicConv(2, 1, 5, stride=1, padding=2, relu=False)
        # fusion operation
        self.conv1x1_fusion = nn.Conv2d(n_feat*2, n_feat*2, kernel_size=1, bias=bias)


    def forward(self, f_event, f_blur):
        ### - blur processing - ###
        fused_feature = f_event + f_blur
        fused_feature = self.trans(fused_feature)
        ## - channel attention - ##
        w_0 = F.adaptive_avg_pool2d(fused_feature, (1,1)).squeeze()
        w_0 = self.F_mlp(w_0)
        w_0 = w_0.reshape(*w_0.shape, 1,1)
        f_blur = f_blur + w_0*f_blur
        ## - spatial attention - ##
        fused_feature2 = fused_feature + self.trans2(fused_feature)
        rgb_att = self.gate_rgb(fused_feature2)
        f_blur = f_blur + rgb_att*f_blur
        ### - dynamic filtering - ###
        kernel_event = self.gernerate_kernel(f_event)
        f_blur = f_blur + self.kconv(f_blur, kernel_event)
        ### - event spatial attention - ###
        sa_event = self.body_sa(f_event)
        e_compressed = self.compress(sa_event)
        e_out = self.spatial_e(e_compressed)
        scale_e = torch.sigmoid(e_out)
        f_event = f_event + scale_e*f_event
        ### - feature fusion - ###
        res = self.conv1x1_fusion(torch.cat((f_event, f_blur), dim=1))
        return res


class Decoder_MS(nn.Module):
    def __init__(self, nf):
        super(Decoder_MS, self).__init__()
        # level 0
        self.deconv_0_1 = conv_resblock_one(8*nf, 8*nf, stride=1)
        # level 1
        self.deconv_1_1 = deconv4x4(8*nf, 4*nf)
        self.deconv_1_2 = conv_resblock_one(8*nf, 4*nf)
        # level 3
        self.deconv_2_1 = deconv4x4(4*nf, 2*nf)
        self.deconv_2_2 = conv_resblock_one(4*nf, 2*nf)
        # pred image block
        self.pred_image0 = conv3x3(8*nf, 3, stride=1)
        self.pred_image1 = conv3x3(4*nf, 3, stride=1)
        self.pred_image2 = conv3x3(2*nf, 3, stride=1)


    def forward(self, x_input):
        # level 0
        x_ = self.deconv_0_1(x_input[2])
        img0 = self.pred_image0(x_)
        # level 1
        x_ = self.deconv_1_1(x_)
        x_in = torch.cat((x_, x_input[1]), dim=1) # test 500
        x_ = self.deconv_1_2(x_in)
        img1 = self.pred_image1(x_)
        # level 2
        x_ = self.deconv_2_1(x_) # 500
        x_in = torch.cat((x_, x_input[0]), dim=1) # test 1GB
        x_ = self.deconv_2_2(x_in) ########### 1GB
        img2 = self.pred_image2(x_)# 1GB
        return [img2, img1, img0]



class EventDeblurNet(nn.Module):
    def __init__(self, num_bins, ES_module=False):
        super(EventDeblurNet, self).__init__()
        unit_bins = num_bins//2
        # number of feature channel
        n_feat = 4
        # accumulated feature number
        n_feat_total = n_feat*9
        # Event encoder
        self.encoder_e = Event_encoder(num_bins, n_feat)
        # blur encoder
        self.encoder_b = Blur_encoder_v3(6, n_feat_total)
        # decoder 1
        self.decoder = Decoder_MS(n_feat_total)
        # scale information
        self.scale = 3
        # feature fusion
        self.ETES_module = Event_selection(self.scale, n_feat)
        # feature fusion
        self.FF_module = nn.ModuleList([DAU_ours_v5(n_feat_total*2**i) for i in range(self.scale)])
        # flag to use ES_module 
        self.ES_module = ES_module

    def forward(self, x_event, x_blur):
        # event encoding
        f_event = self.encoder_e(x_event)
        # blur encoding
        f_blur = self.encoder_b(x_blur)
        # event feature fusion
        f_event_new = []
        for idx in range(len(f_event[0])):
            # generate multi-scale event temporal feature
            f_event_stack = torch.cat((f_event[0][idx][:, None, ...], f_event[1][idx][:, None, ...], \
                                       f_event[2][idx][:, None, ...], f_event[3][idx][:, None, ...], \
                                       f_event[4][idx][:, None, ...] , f_event[5][idx][:, None, ...], \
                                       f_event[6][idx][:, None, ...], f_event[7][idx][:, None, ...], \
                                       f_event[8][idx][:, None, ...]), dim=1) 
            f_event_new.append(f_event_stack)
        # feature fusion
        f_event_new2, viz = self.ETES_module(f_event_new, f_blur)
        x_input = []
        for i in range(self.scale):
            x_input.append(self.FF_module[i](f_event_new2[i], f_blur[i]))
        # decoder
        out_clean = self.decoder(x_input)
        # clean output decoder
        output_clean = []
        for i in range(self.scale):
            output_clean.append(torch.clamp(out_clean[i], 0, 1))
        if self.ES_module:
            return output_clean, viz
        else:
            return output_clean

class EvOurUnknownDeblurNet(nn.Module):
    def __init__(self, num_bins, use_ES_module=True):
        super(EvOurUnknownDeblurNet, self).__init__()
        self.net = EventDeblurNet(num_bins, use_ES_module)
        # use
        self.flag_ES = use_ES_module

    def forward(self, x_event, x_blur):
        B, _, H, W = x_blur.shape
        if self.flag_ES:
            out_clean, viz = self.net(x_event, x_blur)
            return out_clean, viz
        else:
            out_clean = self.net(x_event, x_blur)
            return out_clean