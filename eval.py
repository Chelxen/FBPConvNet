from model import FBPCONVNet
from utils import load_checkpoint, PairedNPYDataset, cmap_convert
from torch.utils.data import DataLoader
import os
import torch
import math
import torchvision
import argparse
import numpy as np
import csv
from skimage.metrics import structural_similarity as ssim
import lpips

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def calculate_psnr(img1, img2, data_range=1.0):
    """Calculate PSNR metric"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(data_range) - 10 * torch.log10(mse)
    return psnr.item()


def calculate_ssim(img1, img2):
    """Calculate SSIM metric"""
    # Convert to numpy array
    img1_np = img1.squeeze().detach().cpu().numpy()
    img2_np = img2.squeeze().detach().cpu().numpy()
    
    # Ensure it's a 2D array
    if len(img1_np.shape) > 2:
        img1_np = img1_np[0] if img1_np.shape[0] == 1 else img1_np
    if len(img2_np.shape) > 2:
        img2_np = img2_np[0] if img2_np.shape[0] == 1 else img2_np
    
    # Normalize to [0, 1] range
    # img1_np = (img1_np - img1_np.min()) / (img1_np.max() - img1_np.min() + 1e-8)
    # img2_np = (img2_np - img2_np.min()) / (img2_np.max() - img2_np.min() + 1e-8)
    
    # Calculate SSIM
    ssim_value = ssim(img1_np, img2_np, data_range=1.0)
    return ssim_value


def eval(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fbp_conv_net = FBPCONVNet().to(device)

    if not (os.path.exists(config.checkpoint_dir) and len(os.listdir(config.checkpoint_dir)) > 0):
        print('load checkpoint unsuccessfully')
        return

    fbp_conv_net, _, _ = load_checkpoint(fbp_conv_net, optimizer=None, checkpoint_dir=config.checkpoint_dir)
    fbp_conv_net.eval()

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)

    print('load testing data')
    # Use DataLoader mode for memory efficiency
    dataset = PairedNPYDataset(config.data_path, config.gt_path, return_filenames=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Create directory if it doesn't exist
    os.makedirs(config.eval_result_dir, exist_ok=True)
    
    # For accumulating metrics
    psnr_values = []
    ssim_values = []
    lpips_values = []
    # Data for saving to CSV
    csv_data = []

    for batch_idx, batch_data in enumerate(dataloader):
        # DataLoader returns tuple: (noisy_batch, orig_batch, filenames) when return_filenames=True
        noisy_batch, orig_batch, filenames = batch_data
        
        # Move to device
        noisy_batch = noisy_batch.to(device)
        orig_batch = orig_batch.to(device)

        y_pred = fbp_conv_net(noisy_batch)
        for j in range(y_pred.shape[0]):
            filename = filenames[j]
            
            # Save pred files using original filename
            pred_image_path = os.path.join(config.eval_result_dir, f'{filename}-pred.jpg')
            pred_npy_path = os.path.join(config.eval_result_dir, f'{filename}-pred.npy')

            # Get prediction and original image tensors for metric calculation
            pred_tensor = y_pred[j].unsqueeze(0)  # Keep batch dimension
            orig_tensor = orig_batch[j].unsqueeze(0)
            
            # Normalize to [0, 1] range for metric calculation
            pred_normalized = (pred_tensor - pred_tensor.min()) / (pred_tensor.max() - pred_tensor.min() + 1e-8)
            orig_normalized = (orig_tensor - orig_tensor.min()) / (orig_tensor.max() - orig_tensor.min() + 1e-8)
            
            # Calculate PSNR
            psnr = calculate_psnr(pred_normalized, orig_normalized)
            psnr_values.append(psnr)
            
            # Calculate SSIM
            ssim_value = calculate_ssim(pred_tensor, orig_tensor)
            ssim_values.append(ssim_value)
            
            # Calculate LPIPS (requires 3 channels, duplicate if single channel)
            if pred_normalized.shape[1] == 1:
                pred_lpips = pred_normalized.repeat(1, 3, 1, 1)
                orig_lpips = orig_normalized.repeat(1, 3, 1, 1)
            else:
                pred_lpips = pred_normalized
                orig_lpips = orig_normalized
            
            # LPIPS requires values in [-1, 1] range
            pred_lpips = pred_lpips * 2.0 - 1.0
            orig_lpips = orig_lpips * 2.0 - 1.0
            
            with torch.no_grad():
                lpips_value = lpips_model(pred_lpips, orig_lpips).item()
            lpips_values.append(lpips_value)
            
            # Save image and npy files
            if config.cmap_convert:
                pred_image = cmap_convert(y_pred[j].squeeze())
                pred_image.save(pred_image_path)
                print('save image:', pred_image_path)
            else:
                torchvision.utils.save_image(y_pred[j].squeeze(), pred_image_path)
                print('save image:', pred_image_path)
            
            # Only save pred npy files
            pred_npy = y_pred[j].clone().detach().cpu().squeeze().numpy()
            np.save(pred_npy_path, pred_npy)
            print('save npy:', pred_npy_path)
            
            # Print metrics
            print(f'{filename}: PSNR={psnr:.4f}, SSIM={ssim_value:.4f}, LPIPS={lpips_value:.4f}')
            
            # Save to CSV data list
            csv_data.append({
                'filename': filename,
                'PSNR': f'{psnr:.6f}',
                'SSIM': f'{ssim_value:.6f}',
                'LPIPS': f'{lpips_value:.6f}'
            })
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    print(f'\nAverage Metrics: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}')
    
    # Save to CSV file
    csv_path = os.path.join(config.eval_result_dir, 'evaluation_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'PSNR', 'SSIM', 'LPIPS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
        
        # Add average metrics row
        writer.writerow({
            'filename': 'Average',
            'PSNR': f'{avg_psnr:.6f}',
            'SSIM': f'{avg_ssim:.6f}',
            'LPIPS': f'{avg_lpips:.6f}'
        })
    
    print(f'\nSaved evaluation metrics to: {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--data_path', type=str, default='/train/90')
    parser.add_argument('--gt_path', type=str, default='/train/gt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_result_dir', type=str, default='./eval_results')
    parser.add_argument('--cmap_convert', type=bool, default=True)
    config = parser.parse_args()
    print(config)
    eval(config)
