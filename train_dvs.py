import torch
from utils.utils import *
from torch.optim import Adam
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from collections import OrderedDict
import datetime
import argparse
import os
from math import ceil
from utils.dataloader import *
from models.OursUnknownDeblur_deblur import UnknwonDeblurNet
import matplotlib.pyplot as plt
import torch.optim as optim


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default=21)
    parser.add_argument('--batch_size', type = int, default=1)
    parser.add_argument('--test_batch_size', type = int, default=1)
    parser.add_argument('--scheduler_mileston', default=[10])
    # training params
    parser.add_argument('--voxel_num_bins', type = int, default=16)
    parser.add_argument('--learning_rate', type = float, default=1e-4)
    parser.add_argument('--mode', type = str, default='train')
    parser.add_argument('--use_ES_module', type=str2bool, default='False')
    # model discription
    parser.add_argument('--model_folder', type=str, default='model_factory')
    parser.add_argument('--model_name', type=str, default='networks_full')
    parser.add_argument('--loss_type', type=str, default='multi_scale')
    # data loading params
    parser.add_argument('--num_threads', type = int, default=2)
    parser.add_argument('--experiment_name', type = str , default='train_blur_unknown_expsoure')
    # tb data
    parser.add_argument('--tb_update_thresh', type = int, default=200)
    parser.add_argument('--tb_folder', type=str, default='./experiments')
    parser.add_argument('--data_dir', type = str, default = '/media/event_video_dataset/dvs_color_blur/')
    parser.add_argument('--train_filename', type = str, default = './filename/dvs_dataset/unknown_exposure/train.txt')
    parser.add_argument('--test_filename_list', type = str , default=['./filename/dvs_dataset/unknown_exposure/test_9-5.txt',
                                                                      './filename/dvs_dataset/unknown_exposure/test_11-3.txt', \
                                                                      './filename/dvs_dataset/unknown_exposure/test_13-1.txt'])
    parser.add_argument('--use_multigpu', type = str2bool, default='True')
    parser.add_argument('--resume_net', type=str2bool, default='False')
    parser.add_argument('--resume_path', type=str, default=None)
    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.tb_iter_cnt = 0
        self.tb_iter_cnt_test = 0
        self.tb_iter_cnt2 = 0
        self.tb_iter_cnt2_test = 0
        self.tb_iter_thresh = args.tb_update_thresh
        self.loss_type = args.loss_type
        self.epochs = args.epochs
        self.batchsize = args.batch_size
        ## tensorboard directory
        tb_path = args.tb_folder + datetime.datetime.now().strftime('%y%m%d-' + args.experiment_name + '/%H%M')
        self.tb = SummaryWriter(tb_path, flush_secs=1)
        # logger
        self._logger = get_logger(tb_path, 'log.txt', 'append')
        self.save_logging_argument(args)
        # train sets
        train_sets = DataLoader_dvs_train(args.data_dir, 'train', args.train_filename, args.voxel_num_bins, training=True)
        # define train data-loader
        self.train_loader = torch.utils.data.DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_threads, drop_last=True, pin_memory=True)
        # test loader dict(GOPRO)
        self.test_loader_dict = []
        for test_name in args.test_filename_list:
            test_sets = DataLoader_dvs_test(args.data_dir, 'test', test_name, args.voxel_num_bins, training=False)
            self.test_loader_dict.append(torch.utils.data.DataLoader(test_sets, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True))
        # define models
        self.model = UnknwonDeblurNet(voxel_num_bins=args.voxel_num_bins, flag_ES=True)
        self.model.initalize(args.model_folder, args.model_name, tb_path)
        # set cuda device
        if torch.cuda.is_available:
            self.model.cuda()
        # multi-gpu use
        if args.use_multigpu:
            self.model.use_multi_gpu()
        # TODO: resuming checkpoint ...
        # optimizer
        params = self.model.get_optimizer_params()
        self.optimizer = Adam(params, lr=args.learning_rate)
        # restoration calculator for evaluation
        self.PSNR_calculator = PSNR()
        self.SSIM_calculator = SSIM()
        # best psnr value for saving..
        self.best_psnr = 0
        # num encoded frame
        self.num_encoded_frame = 2
        # flag ES
        self.flag_ES = args.use_ES_module
        # scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.scheduler_mileston, gamma=0.5)
    
    def save_logging_argument(self, args):
        self._logger.info('**************** Saved argument ******************')
        for arg in vars(args):
            self._logger.info(str(arg) + ': ' + str(getattr(args, arg)))
        self._logger.info('***************************************************')

    def train(self):
        self.model.train()
        for epoch in trange(self.epochs, desc='epoch progress'):
            for _, (sample, time_interval) in enumerate(tqdm(self.train_loader, desc='train progress')):
                self.train_step(sample, time_interval)
            if epoch%2==0:
                # evaluation
                psnr_value, ssim_value = self.test(epoch)
                if self.best_psnr < psnr_value:
                    self.best_psnr = psnr_value
                    self.model.save_model()
            # learning rate scheduling..
            self.scheduler.step()

    def train_step(self, sample, time_interval):
        self.optimizer.zero_grad()
        sample = batch2device(sample)
        # set zero-gradient optimizer
        self.optimizer.zero_grad()
        # set input
        self.model.set_input(sample, time_interval)
        # forward for multi-stage training
        self.model.forward()
        # get loss
        if self.loss_type=='multi_scale':
            loss = self.model.get_multi_scale_loss()
        elif self.loss_type=='single_scale':
            loss = self.model.get_single_loss()
        # backward and step optimizer
        loss.backward()
        self.optimizer.step()
        # loss average meter update
        self.model.update_loss_meters()
        # tb iteration counter
        self.tb_iter_cnt += 1
        if self.batchsize*self.tb_iter_cnt > self.tb_iter_thresh:
            self.log_train_tb()
        del sample
    
    def log_train_tb(self):
        self.tb.add_scalar('train_progress/loss_total', self.model.loss_total_meter.avg, self.tb_iter_cnt2)
        self.tb.add_image('train_blur/blur_image_0', self.model.batch['blur_image_input'][0, :3, ...], self.tb_iter_cnt2)
        self.tb.add_image('train_blur/blur_image_1', self.model.batch['blur_image_input'][0, 3:6, ...], self.tb_iter_cnt2)
        # pred and gt
        self.tb.add_image('train_image/clean_image_est_scale_2', self.model.batch['clean_image_est'][1][0, ...], self.tb_iter_cnt2)
        self.tb.add_image('train_image/clean_image_est_scale_3', self.model.batch['clean_image_est'][0][0, ...], self.tb_iter_cnt2)
        self.tb.add_image('train_image/clean_image_gt', self.model.batch['clean_gt_images'][0, ...], self.tb_iter_cnt2)
        # tensorboard count
        self.tb_iter_cnt2 += 1
        self.tb_iter_cnt = 0
        self.model.reset_loss_meters()

    def channel_value_to_bar_img(self, att_y, time_interval=None):
        x_axis_length  = att_y.shape[0]
        time_unit = int(x_axis_length/9)
        att_x = np.arange(x_axis_length)
        figure = plt.figure()
        plot = figure.add_subplot(111)
        plot.bar(att_x, att_y.squeeze().cpu().detach().numpy())
        if time_interval is not None:
            exp_time, ro_time = time_interval.split('-')
            exp_time = int(exp_time) # 7
            ro_time = int(ro_time) # 5
            unit_time = exp_time+ro_time # 12
            max_idx = int(time_unit+ (x_axis_length-time_unit)  * exp_time/unit_time)   # 30 * 7/12 = 17
            plot.axvline(x=max_idx,color="red",linestyle='--')
            plot.axvline(x=time_unit,color="orange",linestyle='--')
        figure.canvas.draw()
        bar_img=np.array(figure.canvas.renderer._renderer)[:,:,:3]
        plt.close()
        return bar_img.transpose([2,0,1])

    def channel_average_time_value_to_bar_img(self,att_y,time_interval=None):
        x_axis_length = 9
        time_unit = int(att_y.size(0)/9)
        att_x = np.arange(x_axis_length)
        figure = plt.figure()
        plot = figure.add_subplot(111)
        if time_interval is not None:
            x_time, y_time=time_interval.split('-')
            x_time = int(x_time) # 7
            y_time = int(y_time) # 5
            unit_time = x_time+y_time # 12
            att_y = att_y.cpu().detach().numpy()# 72 avg per time_unit # 
            avg_y_list = []
            for i in range(x_axis_length):
                cur_time = att_y[time_unit*i:time_unit*(i+1)].mean()
                avg_y_list.append(cur_time)
            att_y = np.array(avg_y_list)
            plot.bar(att_x,att_y)
            max_idx = int(1 + (x_time/unit_time)*8)   # 1, 7/12 * 8 
            plot.axvline(x=max_idx,color="red",linestyle='--')
            plot.axvline(x=1,color="orange",linestyle='--')
        figure.canvas.draw()
        bar_img=np.array(figure.canvas.renderer._renderer)[:,:,:3]
        plt.close()
        return bar_img.transpose([2,0,1])    

    def test(self, epoch):
        # total evaluation meter
        psnr_meter_clean_total = AverageMeter()
        ssim_meter_clean_total = AverageMeter()
        # blur evaluation meter
        psnr_meter_clean = AverageMeter()
        ssim_meter_clean = AverageMeter()
        l1_meter_clean = AverageMeter()
        # reset evaluation counter
        psnr_meter_clean.reset()
        ssim_meter_clean.reset()
        l1_meter_clean.reset()
        # tensorboard counter
        self.tb_iter_cnt_test = 0 
        self.tb_iter_cnt2_test = 0 
        # model
        self.model.eval()
        with torch.no_grad():
            # evaluation
            self._logger.info('************   ' + 'Evaluation(' + 'epoch: ' + str(epoch) + ')    ************')
            for idx, test_loader in enumerate(self.test_loader_dict):
                for iter_idx, (sample, time_interval) in enumerate(tqdm(test_loader, desc='test progress_' + str(idx))):
                    # go to device
                    sample = batch2device(sample)
                    # self.model.set_input(sample, time_interval)
                    self.model.set_test_input_real(sample, time_interval)
                    # forward for testing
                    self.model.forward_test_real()
                    loss_interp_temp = (((self.model.batch['clean_gt_images'] - self.model.batch['clean_image_est_']) ** 2 + 1e-6) ** 0.5).mean()
                    ssim_var_interp = self.SSIM_calculator(self.model.batch['clean_gt_images'], self.model.batch['clean_image_est_'])
                    psnr_var_interp = self.PSNR_calculator(self.model.batch['clean_gt_images'], self.model.batch['clean_image_est_'])
                    l1_meter_clean.update(loss_interp_temp, 1)
                    # update time-interval meter
                    psnr_meter_clean.update(psnr_var_interp.mean().item(), 1)
                    ssim_meter_clean.update(ssim_var_interp.mean().item(), 1)
                    # update total meter
                    psnr_meter_clean_total.update(psnr_var_interp.mean().item(), 1)
                    ssim_meter_clean_total.update(ssim_var_interp.mean().item(), 1)
                    if self.flag_ES:
                        if iter_idx%500==0:
                            # CA histogram
                            bar_img = self.channel_average_time_value_to_bar_img(self.model.batch['ca_map_list'][0][0],self.model.batch['time_interval'][0]) 
                            time_name='ca_test/{}'.format(self.model.batch['time_interval'][0])
                            self.tb.add_image(time_name, bar_img, self.tb_iter_cnt2_test)  
                            bar_img=self.channel_value_to_bar_img(self.model.batch['ca_map_list'][0][0],self.model.batch['time_interval'][0]) 
                            self.tb.add_image('ca_test/dense', bar_img, self.tb_iter_cnt2_test)
                            # update counter
                            self.tb_iter_cnt2_test += 1
                self.tb.add_scalar('test_progress/DVS/' + time_interval[0] + '/avg_psnr_deblur', psnr_meter_clean.avg, epoch)
                self.tb.add_scalar('test_progress/DVS/' + time_interval[0] + '/avg_ssim_deblur', ssim_meter_clean.avg, epoch)
                self.tb.add_scalar('test_progress/DVS/' + time_interval[0] + '/test_loss_deblur', l1_meter_clean.avg, epoch)
                # logger logging
                self._logger.info(' Time_interval: ' + time_interval[0] + '  PSNR: ' + str(psnr_meter_clean.avg)  + '  SSIM: ' + str(ssim_meter_clean.avg))
                # update total average-meter
                # reset evaluation counter
                psnr_meter_clean.reset()
                ssim_meter_clean.reset()
                l1_meter_clean.reset()
                # tensorboard counter
                self.tb_iter_cnt_test = 0 
                self.tb_iter_cnt2_test = 0 
            self.tb.add_scalar('test_progress/GOPRO/average' +  '/avg_psnr_deblur', psnr_meter_clean_total.avg, epoch)
            self.tb.add_scalar('test_progress/GOPRO/average' +  '/avg_ssim_deblur', ssim_meter_clean_total.avg, epoch)
            self._logger.info(' Total evaluation: ' + '  PSNR: ' + str(psnr_meter_clean_total.avg)  + '  SSIM: ' + str(ssim_meter_clean_total.avg))
        # empty cache !!
        torch.cuda.empty_cache()
        return psnr_meter_clean_total.avg, ssim_meter_clean_total.avg



if __name__=='__main__':
    args = get_argument()
    trainer = Trainer(args)
    if args.mode=='train':
        trainer.train()
    elif args.mode=='test':
        trainer.test(0)