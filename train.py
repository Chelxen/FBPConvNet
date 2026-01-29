from model import FBPCONVNet
import torch
from torch.utils.data import DataLoader
import numpy as np
import math
import argparse
import os
from utils import PairedNPYDataset, load_checkpoint, cmap_convert

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def data_argument(noisy, orig):
    """Apply data augmentation (horizontal and vertical flips)"""
    # flip horizontal
    for i in range(noisy.shape[0]):
        rate = np.random.random()
        if rate > 0.5:
            noisy[i] = noisy[i].flip(2)
            orig[i] = orig[i].flip(2)

    # flip vertical
    for i in range(noisy.shape[0]):
        rate = np.random.random()
        if rate > 0.5:
            noisy[i] = noisy[i].flip(1)
            orig[i] = orig[i].flip(1)
    return noisy, orig


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories if they don't exist
    os.makedirs(config.sample_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    print('load training data')
    # Use DataLoader mode for memory efficiency
    dataset = PairedNPYDataset(config.data_path, config.gt_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    epoch = config.epoch
    grad_max = config.grad_max
    learning_rate = config.learning_rate

    fbp_conv_net = FBPCONVNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(fbp_conv_net.parameters(), lr=learning_rate[0], momentum=config.momentum,
                                weight_decay=1e-8)
    epoch_start = 0

    # load check_point
    if os.path.exists(config.checkpoint_dir) and len(os.listdir(config.checkpoint_dir)) > 0:
        fbp_conv_net, optimizer, epoch_start = load_checkpoint(fbp_conv_net, optimizer, config.checkpoint_dir)

    fbp_conv_net.train()

    print('start training...')
    iteration = 0
    for e in range(epoch_start, epoch):
        # each epoch
        for batch_idx, (noisy_batch, orig_batch) in enumerate(dataloader):
            # Move to device
            noisy_batch = noisy_batch.to(device)
            orig_batch = orig_batch.to(device)
            
            # data augmentation
            noisy_batch, orig_batch = data_argument(noisy_batch, orig_batch)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward Propagation
            y_pred = fbp_conv_net(noisy_batch)

            # save sample images
            iteration += 1
            if iteration % config.sample_step == 0:
                sample_img_path = os.path.join(config.sample_dir, 'epoch-%d-iteration-%d.jpg' % (e + 1, iteration))
                sample_img = cmap_convert(y_pred[0].squeeze())
                sample_img.save(sample_img_path)
                print('save image:', sample_img_path)

            # Compute and print loss
            loss = criterion(y_pred, orig_batch)
            if iteration % 100 == 0:
                print('loss (epoch-%d-iteration-%d) : %f' % (e+1, iteration, loss.item()))

            loss.backward()

            # clip gradient
            torch.nn.utils.clip_grad_value_(fbp_conv_net.parameters(), clip_value=grad_max)

            # Update the parameters
            optimizer.step()

        # adjust learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate[min(e+1, len(learning_rate)-1)]

        # save check_point
        if (e+1) % config.checkpoint_save_step == 0 or (e+1) == config.epoch:
            check_point_path = os.path.join(config.checkpoint_dir, 'epoch-%d.pkl' % (e+1))
            torch.save({'epoch': e+1, 'state_dict': fbp_conv_net.state_dict(), 'optimizer': optimizer.state_dict()},
                       check_point_path)
            print('save checkpoint %s', check_point_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=tuple, default=np.logspace(-2, -3, 20))
    parser.add_argument('--grad_max', type=float, default=0.01)
    parser.add_argument('--batch_size', '--batch', type=int, default=1, dest='batch_size')
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--data_path', type=str, default='/train')
    parser.add_argument('--gt_path', type=str, default='/gt')
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--sample_dir', type=str, default='./samples/')
    parser.add_argument('--checkpoint_save_step', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    config = parser.parse_args()
    print(config)
    main(config)
