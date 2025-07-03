import torch
import os
import scipy.io as scipy_io
import re
import matplotlib.pylab as plt
import PIL.Image as Image
import numpy as np
from numpy.core.multiarray import scalar

# Allow weights_only=True for loading checkpoints
# This is a security measure to prevent loading potentially unsafe objects.
torch.serialization.add_safe_globals([scalar])

# For .mat data
# def load_data(data_path, device, mode):
#     data = scipy_io.loadmat(data_path)
#     noisy = data['lab_d']
#     orig = data['lab_n']
#     noisy = np.transpose(noisy, [3, 2, 0, 1])
#     orig = np.transpose(orig, [3, 2, 0, 1])

#     training_images_count = round(noisy.shape[0]*0.95)
#     if mode == 'train':
#         noisy = torch.tensor(noisy[0:training_images_count]).float().to(device)
#         orig = torch.tensor(orig[0:training_images_count]).float().to(device)
#     elif mode == 'eval':
#         noisy = torch.tensor(noisy[training_images_count:]).float().to(device)
#         orig = torch.tensor(orig[training_images_count:]).float().to(device)
#     else:
#         raise ValueError('mode should be train or test')
#     return noisy, orig

#For .npy data
# def load_data(noisy_dir, orig_dir, device, mode):
#     noisy_files = sorted(os.listdir(noisy_dir))  #Input
#     orig_files = sorted(os.listdir(orig_dir)) #GT

#     noisy_images = [np.load(os.path.join(noisy_dir, f)) for f in noisy_files]
#     orig_images = [np.load(os.path.join(orig_dir, f)) for f in orig_files]

#     noisy_images = np.stack(noisy_images)[:, None, :, :]  
#     orig_images = np.stack(orig_images)[:, None, :, :]  

#     training_images_count = round(noisy_images.shape[0] * 0.8)
    
#     if mode == 'train':
#         noisy = torch.tensor(noisy_images[:training_images_count], device=device, dtype=torch.float)
#         orig = torch.tensor(orig_images[:training_images_count], device=device, dtype=torch.float)
#     elif mode == 'eval':
#         noisy = torch.tensor(noisy_images[training_images_count:], device=device, dtype=torch.float)
#         orig = torch.tensor(orig_images[training_images_count:], device=device, dtype=torch.float)
#     else:
#         raise ValueError('mode should be train or eval')
    
#     return noisy, orig

#For .pt data
def load_data(data_path, device, mode):
    """
    Load data from a .pt file and split it into training and evaluation sets.

    Parameters:
    - data_path: str, the path to the .pt file.
    - device: torch.device, the device to load the tensors onto.
    - mode: str, either 'train' or 'eval' to specify the mode of operation.

    Returns:
    - noisy: torch.Tensor, the noisy images.
    - orig: torch.Tensor, the original images.
    """
    # data = torch.load(data_path)
    data = torch.load(data_path, map_location=device, weights_only=True)
    noisy = data[:, 0, :, :].unsqueeze(1)  # Add a channel dimension
    orig = data[:, 1, :, :].unsqueeze(1)   # Add a channel dimension

    training_images_count = round(noisy.shape[0] * 1)
    if mode == 'train':
        noisy = noisy[:training_images_count].float().to(device)
        orig = orig[:training_images_count].float().to(device)
    elif mode == 'eval':
        noisy = noisy[training_images_count:].float().to(device)
        orig = orig[training_images_count:].float().to(device)
    else:
        raise ValueError('mode should be train or eval')
    
    return noisy, orig


def load_checkpoint(model, optimizer, checkpoint_dir,map_location='cuda:0'):
    if not os.path.exists(checkpoint_dir):
        raise ValueError('checkpoint dir does not exist')

    checkpoint_list = os.listdir(checkpoint_dir)
    if len(checkpoint_list) > 0:

        checkpoint_list.sort(key=lambda x: int(re.findall(r"epoch-(\d+).pkl", x)[0]))

        last_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[-1])
        print('load checkpoint: %s' % last_checkpoint_path)

        # trt to use weights_only=True 
        try:
            model_ckpt = torch.load(last_checkpoint_path, map_location=map_location, weights_only=True)
            print("Successfully loaded checkpoint with weights_only=True")
        except Exception as e:
            print(f"weights_only=True failed: {e}")
            print("Falling back to weights_only=False (less secure but compatible)")
            # if fail, return to weights_only=False
            model_ckpt = torch.load(last_checkpoint_path, map_location=map_location, weights_only=False)

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
