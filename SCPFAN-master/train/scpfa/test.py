import math
import argparse
import yaml
# import utils
import os
from tqdm import tqdm
os.environ["OMP_NUM_THREADS"] = "1"
import utils

from pytorch_msssim import cal_ssim, cal_psnr
parser = argparse.ArgumentParser(description='SCPFA')
# yaml configuration files
parser.add_argument('--exp', type=str, default=None,
                    help='pre-config file for training')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.exp:
        opt = vars(args)
        yaml_args = yaml.load(open(os.path.join("experiments", args.exp, 'config.yml')), Loader=yaml.FullLoader)
        opt.update(yaml_args)
    
    # set visibel gpu
    gpu_ids_str = str(args.gpu_ids).replace('[', '').replace(']', '')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    from datas.utils import create_datasets

    # select active gpu devices
    device = None

    if args.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    # create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(args)

    # definitions of model
    try:
        model = utils.import_module('models.{}_network'.format(args.model)).create_model(args).to(device)
    except Exception:
        raise ValueError('not supported model type! or something')

    # load pretrain
    if args.exp is not None:
        print('load pretrained model: {}!'.format(os.path.join("experiments", args.exp, 'best.pt')))
        ckpt = torch.load(os.path.join("experiments", args.exp, 'best.pt'))
        model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model_state_dict'].items()})
        if True:
            torch.set_grad_enabled(False)
    
            model = model.eval()

            for valid_dataloader in valid_dataloaders:
                avg_psnr, avg_ssim = 0, 0
                name = valid_dataloader['name']
                loader = valid_dataloader['dataloader']
                for lr, hr in tqdm(loader, ncols=80):
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
                    # quantize output to [0, 255]
                    hr = hr.clamp(0, 255)
                    sr = sr.clamp(0, 255)
                    # conver to ycbcr
                    if args.colors == 3:
                        hr_ycbcr = utils.rgb_to_ycbcr(hr)
                        sr_ycbcr = utils.rgb_to_ycbcr(sr)
                        hr = hr_ycbcr[:, 0:1, :, :]
                        sr = sr_ycbcr[:, 0:1, :, :]
                    # crop image for evaluation
                    hr = hr[:, :, args.scale:-args.scale,
                            args.scale:-args.scale]
                    sr = sr[:, :, args.scale:-args.scale,
                            args.scale:-args.scale]
                    # calculate psnr and ssim
                    psnr = cal_psnr(sr, hr, args.exp, name)
                    ssim_res = cal_ssim(sr, hr, args.exp, name)
                    avg_psnr += psnr
                    avg_ssim += ssim_res
                avg_psnr = round(avg_psnr/len(loader), 2)
                avg_ssim = round(avg_ssim/len(loader), 4)
                test_log = '[{}-X{}], PSNR/SSIM: {}/{}\n'.format(
                    name, args.scale, str(avg_psnr), str(avg_ssim))
                print(test_log)