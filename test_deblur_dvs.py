from calendar import c
import torch
import argparse
from PIL import Image
import torchvision.transforms.functional as TF
from skimage import img_as_ubyte
from tqdm import tqdm
from collections import OrderedDict
from utils.utils import *
from utils.visualizer import *


parser = argparse.ArgumentParser()
parser.add_argument('--save_img', type = str2bool, default=True)
parser.add_argument('--save_gt', type = str2bool, default=True)
parser.add_argument('--save_ca', type = str2bool, default=False)
parser.add_argument('--save_input', type = str2bool, default=True)
parser.add_argument('--flag_ES', type = str2bool, default=True)
parser.add_argument('--voxel_num_bins', type = int, default=16)
parser.add_argument('--dataset', type = str, default='dvs')
args = parser.parse_args()


class Tester(object):
    def __init__(self):
        self.reset_index()
    
    def initilialze_synthetic(self, eval_meter, logger):
        self.eval_meter = eval_meter
        self.logger = logger

    def reset_index(self):
        self.global_idx = 0

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        return img

    def test_iter_synthetic(self, test_dir, save_dir, filename, model_name, unit_interval=14):
        # time interval
        time_interval = filename.split('/')[-1].split('.txt')[0].split('_')[1]
        # save dir
        save_input_here = os.path.join(save_dir, 'input_image')
        save_output_here = os.path.join(save_dir, 'output_image')
        save_gt_here = os.path.join(save_dir, 'gt')
        if not os.path.exists(save_input_here):
            os.makedirs(save_input_here)
        if not os.path.exists(save_output_here):
            os.makedirs(save_output_here)
        if not os.path.exists(save_gt_here):
            os.makedirs(save_gt_here)
        # blur image
        blur_image_name = [os.path.join(test_dir, line.split(' ')[0], 'blur_images', line.split(' ')[1], line.split(' ')[2], line.split(' ')[3] +'.png') for line in open(filename)]
        blur_image_name.sort()
        # clean image
        clean_gt_name = [os.path.join(test_dir, line.split(' ')[0], 'clean_gt_images', line.split(' ')[1], line.split(' ')[2], line.split(' ')[3] +'.png') for line in open(filename)]
        clean_gt_name.sort()
        # event data dir
        event_name = [os.path.join(test_dir, line.split(' ')[0], 'event_voxel', line.split(' ')[1], line.split(' ')[2], line.split(' ')[3] +'.npz') for line in open(filename)]
        event_name.sort()
        C_vis_0 = torch.zeros([63])
        for blur_idx in tqdm(range(len(blur_image_name)-1)):
            sample = {}
            # blur images
            B0 = TF.to_tensor(Image.open(blur_image_name[blur_idx]))[None, ...].cuda()
            B1 = TF.to_tensor(Image.open(blur_image_name[blur_idx+1]))[None, ...].cuda()
            sample['blur_image']= torch.cat((B0, B1), dim=1)
            # event data
            EV_tensor_0 = torch.from_numpy(np.load(event_name[blur_idx])["data"]).cuda()
            EV_tensor_1 = torch.from_numpy(np.load(event_name[blur_idx+1])["data"]).cuda()
            sample['voxel_grid']= torch.cat((EV_tensor_0, EV_tensor_1), dim=0)[None, ...]
            # clean gt
            clean_gt = TF.to_tensor(Image.open(clean_gt_name[blur_idx+1]))[None, ...].cuda()
            sample['clean_gt_images'] = clean_gt
            # forward
            deblur_model.set_test_input_real(sample, blur_image_name[blur_idx+1].split('/')[-2])
            deblur_model.forward_test_real()
            C_est = deblur_model.batch['clean_image_est_']
            # to numpy
            C_est_np = self.tensor2numpy(C_est)
            clean_gt_np = self.tensor2numpy(clean_gt)
            psnr_var = self.calc_PSNR(C_est_np, clean_gt_np)
            ssim_var = self.calc_SSIM(C_est_np, clean_gt_np)
            # evaluation metric update
            self.eval_meter['PSNR_interval'].update(psnr_var, 1)
            self.eval_meter['SSIM_interval'].update(ssim_var, 1)
            self.eval_meter['PSNR_total'].update(psnr_var, 1)
            self.eval_meter['SSIM_total'].update(ssim_var, 1)
            if args.save_img:
                save_img(os.path.join(save_output_here, str(self.global_idx).zfill(5) + '.png'), C_est_np)
            if args.save_gt:
                save_img(os.path.join(save_gt_here, str(self.global_idx).zfill(5) + '.png'), clean_gt_np)
            if args.save_input:
                B1 = torch.clamp(B1, 0, 1)
                B1 = B1.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()
                inp_img = img_as_ubyte(B1)
                save_img(os.path.join(save_input_here, str(self.global_idx).zfill(5) + '.png'), inp_img)
            if args.save_ca:
                temp = deblur_model.batch['ca_map_list'][0].detach().cpu().squeeze()
                C_vis_0 += temp
                _ = deblur_model.batch['clean_image_est_']
                del temp
            self.global_idx += 1
        C_vis_0_total = C_vis_0 / (blur_idx+1)
        out_val = channel_average_time_value_to_bar_img(C_vis_0_total, time_interval=time_interval)
        Image.fromarray(out_val).save(os.path.join(save_dir, 'exposure_time.png'))  
        self.logger.info('Interval eval: time interval: ' + time_interval +  '  Avg psnr: ' + str(self.eval_meter['PSNR_interval'].avg))
        self.logger.info('Interval eval: Time interval: ' + time_interval +  '  Avg ssim: ' + str(self.eval_meter['SSIM_interval'].avg))
        self.eval_meter['PSNR_interval'].reset()
        self.eval_meter['SSIM_interval'].reset()

    def test_iter_real_shot(self, test_dir, filename, model_name):
        # save dir
        save_input_here = os.path.join('./output', 'real_shot', model_name, filename, 'input_image')
        save_output_here = os.path.join('./output', 'real_shot', model_name, filename, 'output_image')
        if not os.path.exists(save_input_here):
            os.makedirs(save_input_here)
        if not os.path.exists(save_output_here):
            os.makedirs(save_output_here)
        # blur image
        blur_data_dir = os.path.join(test_dir, 'frames')
        blur_image_name = [os.path.join(blur_data_dir, blur_name) for blur_name in os.listdir(blur_data_dir)]
        blur_image_name.sort()
        # event data dir
        event_data_dir = os.path.join(test_dir, 'event_voxel_grid_bin16')
        event_name = [os.path.join(event_data_dir, event_name) for event_name in os.listdir(event_data_dir)]
        event_name.sort()
        C_vis_0 = np.zeros([63])
        for blur_idx in tqdm(range(len(blur_image_name)-1)):
            sample = {}
            # blur images
            B0 = TF.to_tensor(Image.open(blur_image_name[blur_idx]))[None, ...].cuda()
            B1 = TF.to_tensor(Image.open(blur_image_name[blur_idx+1]))[None, ...].cuda()
            sample['blur_image']= torch.cat((B0, B1), dim=1)
            # event data
            EV_tensor_0 = torch.from_numpy(np.load(event_name[blur_idx])["data"]).cuda()
            EV_tensor_1 = torch.from_numpy(np.load(event_name[blur_idx+1])["data"]).cuda()
            sample['voxel_grid']= torch.cat((EV_tensor_0, EV_tensor_1), dim=0)[None, ...]
            # forward
            deblur_model.set_test_input_real_shot(sample)
            deblur_model.forward_test_real()
            C_est = deblur_model.batch['clean_image_est_']
            if args.save_img:
                C_est = torch.clamp(C_est, 0, 1)
                restored = C_est.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()
                restored_img = img_as_ubyte(restored)
                save_img(os.path.join(save_output_here, str(self.global_idx).zfill(5) + '.png'), restored_img)
            if args.save_input:
                B1 = torch.clamp(B1, 0, 1)
                B1 = B1.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()
                inp_img = img_as_ubyte(B1)
                save_img(os.path.join(save_input_here, str(self.global_idx).zfill(5) + '.png'), inp_img)
            if args.save_ca:
                temp = deblur_model.batch['ca_map_list'][0].squeeze().detach().numpy()
                C_vis_0 += temp
            self.global_idx += 1
        C_vis_0_total = C_vis_0 / (blur_idx+1)
        out_val = channel_average_time_value_to_bar_img(C_vis_0_total, time_interval=time_interval)
        Image.fromarray(out_val).save(os.path.join(save_dir, filename + '.png'))  
        
    def ca_visual(self, test_dir, filename, model_name, time_interval):
        # real shot save dir
        save_dir = os.path.join('./output', 'real_shot', model_name, filename)
        # save dir
        save_input_here = os.path.join(save_dir, 'input_image')
        save_output_here = os.path.join(save_dir, 'output_image')
        if not os.path.exists(save_input_here):
            os.makedirs(save_input_here)
        if not os.path.exists(save_output_here):
            os.makedirs(save_output_here)
        # blur image
        blur_data_dir = os.path.join(test_dir, 'frames')
        blur_image_name = [os.path.join(blur_data_dir, blur_name) for blur_name in os.listdir(blur_data_dir)]
        blur_image_name.sort()
        # event data dir
        event_data_dir = os.path.join(test_dir, 'event_voxel_grid_bin16')
        event_name = [os.path.join(event_data_dir, event_name) for event_name in os.listdir(event_data_dir)]
        event_name.sort()
        # for visualization
        C_vis_0 = torch.zeros([63]).cuda()
        for blur_idx in tqdm(range(len(blur_image_name)-1)):
            with torch.no_grad():
                sample = {}
                # blur images
                B0 = TF.to_tensor(Image.open(blur_image_name[blur_idx]))[None, ...].cuda()
                B1 = TF.to_tensor(Image.open(blur_image_name[blur_idx+1]))[None, ...].cuda()
                sample['blur_image']= torch.cat((B0, B1), dim=1)
                # event data
                EV_tensor_0 = torch.from_numpy(np.load(event_name[blur_idx])["data"]).cuda()
                EV_tensor_1 = torch.from_numpy(np.load(event_name[blur_idx+1])["data"]).cuda()
                sample['voxel_grid']= torch.cat((EV_tensor_0, EV_tensor_1), dim=0)[None, ...]
                # forward
                deblur_model.set_test_input_real_shot(sample)
                deblur_model.forward_test_real()
                
                temp = deblur_model.batch['ca_map_list'][0].squeeze()
                C_vis_0 += temp
                _ = deblur_model.batch['clean_image_est_']
                self.global_idx += 1
        C_vis_0_total = C_vis_0 / (blur_idx+1)
        out_val = channel_average_time_value_to_bar_img(C_vis_0_total, time_interval=time_interval)
        Image.fromarray(out_val).save(os.path.join(save_dir, filename + '.png'))  


if __name__=='__main__':
    # model name specification
    model_name = 'Ours'
    # test mode spec
    test_mode = 'synthetic_dvs'
    if test_mode == 'real_shot':
        # test dir
        test_dir = '/media/mnt3/event_dataset/real_blurry_event_dataset/unpacked_file/'
        filename_list = ['dvSave-2022_01_28_15_32_58']
        filename_list.sort()
    elif test_mode == 'synthetic_dvs':
        # test dir
        test_dir = '/media/mnt3/event_dataset/real_blurry_event_dataset/blur_dataset/unknown_exposure_public/test'
        filename_list = ['./filename/dvs_dataset/test_9-5.txt', './filename/dvs_dataset/test_11-3.txt', './filename/dvs_dataset/test_13-1.txt']
        eval_meter = {}
        eval_meter['PSNR_total'] = AverageMeter()
        eval_meter['SSIM_total'] = AverageMeter()
        eval_meter['PSNR_interval'] = AverageMeter()
        eval_meter['SSIM_interval'] = AverageMeter()
    from models.DeblurNet import UnknwonDeblurNet
    deblur_model = UnknwonDeblurNet(voxel_num_bins=args.voxel_num_bins, flag_ES=True)
    deblur_model.initalize('model_factory', 'networks_light', tb_path=None)
    deblur_model.cuda()
    pretrained_base_dir = './pretrained/Ours_light_final/saved_model/best_model.pth'
    ckpt = torch.load(pretrained_base_dir)
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:]
        new_state_dict[name] = v
    deblur_model.load_model(new_state_dict)
    test_obj = Tester()
    if test_mode=='real_shot':
        for filename in filename_list:
            test_dir_new = os.path.join(test_dir, filename)
            # test_obj.ca_visual(test_dir_new, filename, model_name)
            test_obj.test_iter_real_shot(test_dir_new, filename, model_name)
    elif test_mode == 'synthetic_dvs':
        save_base = os.path.join('./output', 'synthetic_dvs', model_name)
        if not os.path.exists(save_base):
            os.makedirs(save_base)
        logger = get_logger(save_base, model_name + '-' + args.dataset + '.txt' , 'write')
        logger.info('------------------ evaluation logs ------------------')
        logger.info('model name : '  + model_name)
        for filename in filename_list:
            test_obj = Tester()
            time_interval = filename.split('/')[-1].split('.txt')[0].split('_')[1]
            save_dir = os.path.join(save_base, time_interval)
            test_obj.initilialze_synthetic(eval_meter, logger)
            # test_obj.ca_visual(test_dir, filename, model_name, time_interval)
            test_obj.test_iter_synthetic(test_dir, save_dir, filename, model_name)
        test_obj.logger.info('----------------------------------------------')
        test_obj.logger.info('Total evaluation: ' +  'avg psnr: ' + str(test_obj.eval_meter['PSNR_total'].avg))
        test_obj.logger.info('Total evaluation: ' +  'avg ssim: ' + str(test_obj.eval_meter['SSIM_total'].avg))
        test_obj.eval_meter['PSNR_total'].reset()
        test_obj.eval_meter['SSIM_total'].reset()