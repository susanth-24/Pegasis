import torch
import kornia.metrics as metrics
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int,
    max_val: float = 100.0,
    eps: float = 1e-12,
    reduction: str = 'mean',
    padding: str = 'same',
) -> torch.Tensor:
    '''
    do "!pip install kornia" to use this function
    Structural similarity index measure alias SSIM loss is being used for calculating the loss
    This function takes in predicted, and ground truth depth image,
    window size 11 is recommended, smaller size can take in more noice, larger size can avoid important features.
    '''
    ssim_map = metrics.ssim(img1, img2, window_size, max_val, eps, padding)

    loss = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    elif reduction == 'none':
        pass

    return loss