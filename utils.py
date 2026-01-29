import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import scipy.io as scipy_io
import re
import matplotlib.pylab as plt
import PIL.Image as Image
import numpy as np


class NPYDataset(Dataset):
    """
    Dataset class for loading npy files on-demand
    This is more memory-efficient than loading all data at once
    """
    def __init__(self, data_path, return_filenames=False):
        """
        Args:
            data_path: Path to directory containing npy files
            return_filenames: Whether to return filenames along with data
        """
        if not os.path.isdir(data_path):
            raise ValueError(f'data_path {data_path} is not a valid directory')
        
        self.data_files = sorted(glob.glob(os.path.join(data_path, '*.npy')))
        
        if len(self.data_files) == 0:
            raise ValueError(f'No npy files found in {data_path}')
        
        print(f'Found {len(self.data_files)} npy files in {data_path}')
        
        self.return_filenames = return_filenames
        self.filenames = [os.path.splitext(os.path.basename(f))[0] for f in self.data_files]
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load data on-demand
        data = np.load(self.data_files[idx])
        
        # Ensure correct dimensions (assume 2D image, add channel dimension)
        if len(data.shape) == 2:
            data = data[np.newaxis, :, :]  # (1, H, W)
        elif len(data.shape) == 3:
            # Already has channel dimension
            pass
        else:
            raise ValueError(f'Unexpected data shape: {data.shape}')
        
        # Convert to torch tensor
        data_tensor = torch.tensor(data).float()
        
        if self.return_filenames:
            return data_tensor, self.filenames[idx]
        return data_tensor


class PairedNPYDataset(Dataset):
    """
    Dataset class for loading paired npy files (noisy and ground truth)
    """
    def __init__(self, noisy_path, gt_path, return_filenames=False):
        """
        Args:
            noisy_path: Path to directory containing noisy npy files
            gt_path: Path to directory containing ground truth npy files
            return_filenames: Whether to return filenames along with data
        """
        if not os.path.isdir(noisy_path):
            raise ValueError(f'noisy_path {noisy_path} is not a valid directory')
        if not os.path.isdir(gt_path):
            raise ValueError(f'gt_path {gt_path} is not a valid directory')
        
        noisy_files = sorted(glob.glob(os.path.join(noisy_path, '*.npy')))
        gt_files = sorted(glob.glob(os.path.join(gt_path, '*.npy')))
        
        if len(noisy_files) == 0:
            raise ValueError(f'No npy files found in {noisy_path}')
        if len(gt_files) == 0:
            raise ValueError(f'No npy files found in {gt_path}')
        
        # Match files by name
        noisy_dict = {os.path.basename(f): f for f in noisy_files}
        gt_dict = {os.path.basename(f): f for f in gt_files}
        
        self.pairs = []
        self.filenames = []
        for filename in sorted(noisy_dict.keys()):
            if filename in gt_dict:
                self.pairs.append((noisy_dict[filename], gt_dict[filename]))
                self.filenames.append(os.path.splitext(filename)[0])
            else:
                print(f'Warning: {filename} found in noisy_path but not in gt_path')
        
        if len(self.pairs) == 0:
            raise ValueError('No matching file pairs found between noisy_path and gt_path')
        
        print(f'Found {len(self.pairs)} matching file pairs')
        self.return_filenames = return_filenames
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        noisy_file, gt_file = self.pairs[idx]
        
        # Load data on-demand
        noisy_data = np.load(noisy_file)
        gt_data = np.load(gt_file)
        
        # Ensure correct dimensions
        if len(noisy_data.shape) == 2:
            noisy_data = noisy_data[np.newaxis, :, :]  # (1, H, W)
        elif len(noisy_data.shape) == 3:
            pass
        else:
            raise ValueError(f'Unexpected noisy data shape: {noisy_data.shape}')
        
        if len(gt_data.shape) == 2:
            gt_data = gt_data[np.newaxis, :, :]  # (1, H, W)
        elif len(gt_data.shape) == 3:
            pass
        else:
            raise ValueError(f'Unexpected gt data shape: {gt_data.shape}')
        
        # Convert to torch tensors
        noisy_tensor = torch.tensor(noisy_data).float()
        gt_tensor = torch.tensor(gt_data).float()
        
        if self.return_filenames:
            return noisy_tensor, gt_tensor, self.filenames[idx]
        return noisy_tensor, gt_tensor


def load_data(data_path, device, return_filenames=False, use_dataloader=False, batch_size=1, shuffle=False):
    """
    Load data from npy files
    
    Args:
        data_path: Path to directory containing npy files
        device: Device to move data to (only used if use_dataloader=False)
        return_filenames: Whether to return filenames
        use_dataloader: If True, return DataLoader (memory-efficient). If False, load all data at once (old behavior)
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data in DataLoader
    
    Returns:
        If use_dataloader=True: DataLoader object
        If use_dataloader=False: Tensor on device (old behavior for backward compatibility)
    """
    if use_dataloader:
        # Memory-efficient: return DataLoader
        dataset = NPYDataset(data_path, return_filenames=return_filenames)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return dataloader
    else:
        # Old behavior: load all data at once (for backward compatibility)
        # This is memory-intensive but may be fine for small datasets
        if not os.path.isdir(data_path):
            raise ValueError(f'data_path {data_path} is not a valid directory')
        
        data_files = sorted(glob.glob(os.path.join(data_path, '*.npy')))
        
        if len(data_files) == 0:
            raise ValueError(f'No npy files found in {data_path}')
        
        print(f'Found {len(data_files)} npy files in {data_path}')
        
        filenames = [os.path.splitext(os.path.basename(f))[0] for f in data_files]
        
        # Load all files
        data_list = []
        
        for data_file in data_files:
            data = np.load(data_file)
            
            # Ensure correct dimensions
            if len(data.shape) == 2:
                data = data[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
            elif len(data.shape) == 3:
                data = data[np.newaxis, :, :, :]  # (1, C, H, W)
            
            data_list.append(data)
        
        # Concatenate all data
        data_tensor = np.concatenate(data_list, axis=0)
        
        # Convert to torch tensor and move to device
        data_tensor = torch.tensor(data_tensor).float().to(device)
        
        if return_filenames:
            return data_tensor, filenames
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
