import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_versus(lr, sr, hr):
    fig = plt.figure(figsize=(12, 5))
    fig.add_subplot(131)

    plt.imshow(torch.clamp(lr, min=0.0, max=255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    plt.axis('off')
    plt.title("LR")

    fig.add_subplot(132)
    plt.imshow(torch.clamp(sr, min=0.0, max=255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    plt.axis('off')
    plt.title("Super Resolution")

    fig.add_subplot(133)
    plt.imshow(torch.clamp(hr, min=0.0, max=255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    plt.axis('off')
    plt.title("Ground Truth HR")

    return fig

