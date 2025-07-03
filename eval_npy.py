from model import FBPCONVNet
from utils import load_checkpoint, load_data, cmap_convert, rsnr
import os
import torch
import math
import torchvision
import argparse
import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import statistics 
import time


import lpips

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
This files is used to evaluate the model on test dataset.
The evaluation metrics include SSIM, PSNR, LPIPS.

The results will be saved as .npy and .png format in the eval_result_dir.

Chelsen Fang
2025-07-03
'''

ssim_list = []
psnr_list = []
lpips_list = []
time_list = [] 


def eval(config):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ============= LPIPS init =============
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    fbp_conv_net = FBPCONVNet().to(device)

    start_time = time.time()  # start timer

    if not (os.path.exists(config.checkpoint_dir) and len(os.listdir(config.checkpoint_dir)) > 0):
        print('load checkpoint unsuccessfully')
        return

    fbp_conv_net, _, _ = load_checkpoint(fbp_conv_net, optimizer=None, checkpoint_dir=config.checkpoint_dir)
    fbp_conv_net.eval()

    print('load testing data')
    # noisy, orig = load_data(config.data_path, device, mode='eval')
    noisy, orig = load_data(config.data_path, device=device, mode='train')
    print("Loaded data shapes:", noisy.shape, orig.shape)


    if not os.path.exists(config.eval_result_dir):
        os.mkdir(config.eval_result_dir)

    for i in range(math.ceil(noisy.shape[0]/config.batch_size)):
        i_start = i
        i_end = min(i+config.batch_size, noisy.shape[0])
        noisy_batch = noisy[i_start:i_end]
        orig_batch = orig[i_start:i_end]

        y_pred = fbp_conv_net(noisy_batch)
        for j in range(y_pred.shape[0]):
            noisy_image_path = os.path.join(config.eval_result_dir, '%d-noisy.jpg' % (i * config.batch_size + j + 1))
            pred_image_path = os.path.join(config.eval_result_dir, '%d-pred.jpg' % (i*config.batch_size+j+1))
            orig_image_path = os.path.join(config.eval_result_dir, '%d-orig.jpg' % (i*config.batch_size+j+1))
            idx = i_start + j
            
            if config.cmap_convert:
                noisy_image = cmap_convert(noisy_batch[j].squeeze())
                noisy_image.save(noisy_image_path)
                print('save image:', noisy_image_path)

                pred_image = cmap_convert(y_pred[j].squeeze())
                pred_image.save(pred_image_path)
                print('save image:', pred_image_path)

                orig_image = cmap_convert(orig_batch[j].squeeze())
                orig_image.save(orig_image_path)
                print('save image:', orig_image_path)

                SNR = rsnr(np.array(pred_image), np.array(orig_image))
                print('%d-pred.jpg SNR:%f' % (i * config.batch_size + j + 1, SNR))

            else:
                torchvision.utils.save_image(noisy_batch[j].squeeze(), noisy_image_path)
                print('save image:', noisy_image_path)
                torchvision.utils.save_image(y_pred[j].squeeze(), pred_image_path)
                print('save image:', pred_image_path)
                torchvision.utils.save_image(orig_batch[j].squeeze(), orig_image_path)
                print('save image:', orig_image_path)

                pred_image = y_pred[j].clone().detach().cpu().squeeze()
                orig_image = orig_batch[j].clone().detach().cpu().squeeze()
                SNR = rsnr(np.array(pred_image), np.array(orig_image))
                print('%d-pred.jpg SNR:%f' % (i * config.batch_size + j + 1, SNR))
            
            # --------------------
            # save as npy files
            # --------------------
            noisy_path_npy = os.path.join(config.eval_result_dir, f'{idx+1}-noisy.npy')
            pred_path_npy  = os.path.join(config.eval_result_dir, f'{idx+1}-pred.npy')
            orig_path_npy  = os.path.join(config.eval_result_dir, f'{idx+1}-orig.npy')
            
            np.save(noisy_path_npy, noisy_batch[j].squeeze().detach().cpu().numpy())
            np.save(pred_path_npy,  y_pred[j].squeeze().detach().cpu().numpy())
            np.save(orig_path_npy,  orig_batch[j].squeeze().detach().cpu().numpy())

            print(f'Saved: {noisy_path_npy}, {pred_path_npy}, {orig_path_npy}')            

            pred_data = y_pred[j].clone().detach().cpu().numpy()
            orig_data = orig_batch[j].clone().detach().cpu().numpy()

            if len(pred_data.shape) == 3:
                if pred_data.shape[0] == 1:
                    pred_data = pred_data[0]  
                    orig_data = orig_data[0]

            ssim_value = ssim(pred_data, orig_data, data_range=1)
            # ssim_value = ssim(pred_data, orig_data, data_range=orig_data.max() - orig_data.min())
            psnr_value = psnr(pred_data, orig_data, data_range=orig_data.max() - orig_data.min())

            #     LPIPS requires 3 channels [N, 3, H, W], if input is grayscale, we need to repeat the channel and normalize
            pred_tensor = y_pred[j].detach()
            orig_tensor = orig_batch[j].detach()

            # 如果数据在 [0, 1]，可以直接 repeat 三通道
            if pred_tensor.shape[1:] == torch.Size([1, pred_tensor.shape[2], pred_tensor.shape[2]]):
                pred_tensor = pred_tensor.repeat(3,1,1)
                orig_tensor = orig_tensor.repeat(3,1,1)

            pred_tensor = pred_tensor.unsqueeze(0).to(device)
            orig_tensor = orig_tensor.unsqueeze(0).to(device)

            lpips_value = lpips_fn(pred_tensor, orig_tensor).item()

            ssim_list.append(ssim_value)
            psnr_list.append(psnr_value)
            lpips_list.append(lpips_value)

            print(f'[{idx+1}-pred] SSIM:{ssim_value:.4f}, PSNR:{psnr_value:.4f}, LPIPS:{lpips_value:.4f}')

    # ============= Caculate mean metrics  =============
    mean_ssim  = sum(ssim_list)  / len(ssim_list)
    mean_psnr  = sum(psnr_list)  / len(psnr_list)
    mean_lpips = sum(lpips_list) / len(lpips_list)

    std_ssim   = statistics.stdev(ssim_list)
    std_psnr   = statistics.stdev(psnr_list)
    std_lpips  = statistics.stdev(lpips_list)

    print(f'[Overall] SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}')
    print(f'[Overall] PSNR: {mean_psnr:.4f} ± {std_psnr:.4f}')
    print(f'[Overall] LPIPS: {mean_lpips:.4f} ± {std_lpips:.4f}')

    # ============= Timer =============
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / len(ssim_list)

    print(f'[Time] Total Time: {total_time:.2f} sec')
    print(f'[Time] Avg Time per Image: {avg_time_per_image:.2f} sec')


    # ============= Save logs as txt =============
    metrics_txt = os.path.join(config.eval_result_dir, 'metrics.txt')
    with open(metrics_txt, 'w') as f:
        f.write(f'SSIM:  {mean_ssim:.4f} ± {std_ssim:.4f}\n')
        f.write(f'PSNR:  {mean_psnr:.4f} ± {std_psnr:.4f}\n')
        f.write(f'LPIPS: {mean_lpips:.4f} ± {std_lpips:.4f}\n')
        f.write(f'Total Time: {total_time:.2f} sec\n')
        f.write(f'Avg Time per Image: {avg_time_per_image:.2f} sec\n')    

    print('Metrics saved to', metrics_txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')  
    parser.add_argument('--data_path', type=str, default='test_dataset_path.pt')  
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_result_dir', type=str, default='path_to_save_your_eval_results') 
    parser.add_argument('--cmap_convert', type=bool, default=True)
    # parser.add_argument('--data_input', type=str, default='/mnt/4b9cdae1-f581-4f95-aa23-5b45c0bdf521/changsheng/pytorch-template-medical-image-restoration/dataset/AAPM/Low_dose')    
    # parser.add_argument('--data_gt', type=str, default='/mnt/4b9cdae1-f581-4f95-aa23-5b45c0bdf521/changsheng/pytorch-template-medical-image-restoration/dataset/AAPM/High_dose')
    config = parser.parse_args()
    print(config)
    eval(config)