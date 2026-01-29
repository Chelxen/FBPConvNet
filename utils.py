import torch
import os
import glob
import scipy.io as scipy_io
import re
import matplotlib.pylab as plt
import PIL.Image as Image
import numpy as np


def load_data(data_path, device):
    """
    从npy文件加载数据
    data_path: 数据文件夹路径，直接读取该路径下的所有npy文件
    """
    # 获取所有npy文件并排序
    if not os.path.isdir(data_path):
        raise ValueError(f'data_path {data_path} is not a valid directory')
    
    data_files = sorted(glob.glob(os.path.join(data_path, '*.npy')))
    
    if len(data_files) == 0:
        raise ValueError(f'No npy files found in {data_path}')
    
    print(f'Found {len(data_files)} npy files in {data_path}')
    
    # 加载所有文件
    data_list = []
    
    for data_file in data_files:
        data = np.load(data_file)
        
        # 确保数据维度正确 (假设是2D图像，需要添加batch和channel维度)
        if len(data.shape) == 2:
            data = data[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        elif len(data.shape) == 3:
            data = data[np.newaxis, :, :, :]  # (1, C, H, W)
        
        data_list.append(data)
    
    # 合并所有数据
    data_tensor = np.concatenate(data_list, axis=0)
    
    # 转换为torch tensor并移动到设备
    data_tensor = torch.tensor(data_tensor).float().to(device)
    
    return data_tensor


def load_checkpoint(model, optimizer, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        raise ValueError('checkpoint dir does not exist')

    checkpoint_list = os.listdir(checkpoint_dir)
    if len(checkpoint_list) > 0:

        checkpoint_list.sort(key=lambda x: int(re.findall(r"epoch-(\d+).pkl", x)[0]))

        last_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[-1])
        print('load checkpoint: %s' % last_checkpoint_path)

        model_ckpt = torch.load(last_checkpoint_path)
        model.load_state_dict(model_ckpt['state_dict'])

        if optimizer:
            optimizer.load_state_dict(model_ckpt['optimizer'])

        epoch = model_ckpt['epoch']
        return model, optimizer, epoch


def cmap_convert(image_tensor):
    image = image_tensor.detach().cpu().clone().numpy().squeeze()
    # subtract the minimum and divide by the maximum, is it reasonable for medical image processing
    image = image - image.min()
    image = image / image.max()
    cmap_viridis = plt.get_cmap('viridis')
    image = cmap_viridis(image)
    image = Image.fromarray((image * 255).astype(np.uint8)).convert('L')
    return image


def rsnr(rec, oracle):
    "regressed SNR"
    sumP = sum(oracle.reshape(-1))
    sumI = sum(rec.reshape(-1))
    sumIP = sum(oracle.reshape(-1) * rec.reshape(-1) )
    sumI2 = sum(rec.reshape(-1)**2)
    A = np.matrix([[sumI2, sumI], [sumI, oracle.size]])
    b = np.matrix([[sumIP], [sumP]])
    c = np.linalg.inv(A)*b #(A)\b
    rec = c[0, 0]*rec+c[1, 0]
    err = sum((oracle.reshape(-1)-rec.reshape(-1))**2)
    SNR = 10.0*np.log10(sum(oracle.reshape(-1)**2)/err)

    if np.isnan(SNR):
        SNR = 0.0
    return SNR
