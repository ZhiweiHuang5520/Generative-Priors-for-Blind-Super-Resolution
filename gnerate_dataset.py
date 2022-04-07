import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters, measurements, interpolation
import glob
from scipy.io import savemat
import os
from PIL import Image
import torch
import torch.nn.functional as F
import argparse

# Function for centering a kernel
def kernel_shift(kernel, sf):
    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The idea kernel center
    # for image blurred by filters.correlate
    # wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))
    # for image blurred by F.conv2d. They are the same after kernel.flip([0,1])
    wanted_center_of_mass = (np.array(kernel.shape) - sf) / 2.

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)

# Function for generating one fixed kernel
def gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise):
    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2]);
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 + 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    kernel = raw_kernel_centered / np.sum(raw_kernel_centered)

    return kernel

# Function for generating one random kernel
def gen_kernel_random(k_size, scale_factor, min_var, max_var, noise_level):
    lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    kernel = gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise)

    return kernel

def degradation(input, kernel, scale_factor, noise_im, device=torch.device('cuda')):
    # preprocess image and kernel
    input = torch.from_numpy(input).type(torch.FloatTensor).to(device).unsqueeze(0).permute(3, 0, 1, 2)
    input = F.pad(input, pad=(kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2),
                  mode='circular')
    kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)

    # blur
    output = F.conv2d(input, kernel)
    output = output.permute(2, 3, 0, 1).squeeze(3).cpu().numpy()

    # down-sample
    output = output[::scale_factor[0], ::scale_factor[1], :]

    # add AWGN noise
    output += np.random.normal(0, np.random.uniform(0, noise_im), output.shape)

    return output

# Function for generating one random kernel
def gen_kernel_random(k_size, scale_factor, min_var, max_var, noise_level):
    lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    kernel = gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise)

    return kernel

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def generate_dataset(images_path, out_path_im, out_path_ker, k_size, scale_factor, min_var, max_var, noise_ker,
                     noise_im):
    os.makedirs(out_path_im, exist_ok=True)
    os.makedirs(out_path_ker, exist_ok=True)

    # Load images, downscale using kernels and save
    files_source = glob.glob(images_path)
    files_source.sort()
    for i, path in enumerate(files_source):
        print(path)

        im = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.

        im = modcrop(im, scale_factor[0])
        kernel = gen_kernel_random(k_size, scale_factor, min_var, max_var, noise_ker)

        lr = degradation(im, kernel, scale_factor, noise_im,
                         device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        savemat('%s/%s.mat' % (out_path_ker, os.path.splitext(os.path.basename(path))[0]), {'Kernel': kernel})
        plt.imsave('%s/%s.png' % (out_path_im, os.path.splitext(os.path.basename(path))[0]),
                   np.clip(lr, 0, 1), vmin=0, vmax=1)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sf', type=int, default=2, help='scale factor: 2, 3, 4, 8')
    parser.add_argument('--split', type=str, default="val", help='train / val / test')
    parser.add_argument('--dataset', type=str, default='DF2K', help='dataset: Set5, Set14, BSD100, Urban100, DF2K')
    parser.add_argument('--noise_ker', type=float, default=0, help='noise on kernel, e.g. 0.4')
    parser.add_argument('--noise_im', type=float, default=0, help='noise on LR image, e.g. 10/255=0.039')
    opt = parser.parse_args()

    images_path = 'datasets/{}/{}/HR/*.png'.format(opt.dataset, opt.split)
    out_path_im = 'datasets/{}/{}/lr_x{}'.format(opt.dataset, opt.split, opt.sf)
    out_path_ker = 'datasets/{}/{}/gt_k_x{}'.format(opt.dataset, opt.split, opt.sf)

    min_var = 0.175 * opt.sf
    max_var = min(2.5 * opt.sf, 10)
    k_size = np.array([min(opt.sf * 4 + 3, 21), min(opt.sf * 4 + 3, 21)]) # 11x11, 15x15, 19x19, 21x21 for x2, x3, x4, x8
    generate_dataset(images_path, out_path_im, out_path_ker, k_size, np.array([opt.sf, opt.sf]), min_var, max_var,
                     opt.noise_ker, opt.noise_im)


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()
    sys.exit()