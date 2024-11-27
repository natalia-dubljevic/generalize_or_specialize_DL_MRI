import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_root_mse
from typing import Tuple
import torch
"""
"""


def SSIM(rec, target):
    """
    Get SSIM metrics for a reconstruction / target pair of slices
    """
    data_range = target.max() - target.min()
    ssim = structural_similarity(target, rec, data_range=data_range)

    return ssim


def pSNR(rec, target):
    """
    Get pSNR metrics for a reconstruction / target pair of slices
    """
    data_range = target.max() - target.min()
    psnr = peak_signal_noise_ratio(target, rec, data_range=data_range)

    return psnr


def NAE(rec, target, return_map=False):
    """
    Calculate the normalized absolute error for a reconstruction / target
    pair of slices. Metric is normalized by mean of target. Can return the 
    mean NAE, or a map of values.
    """
    map = np.abs(np.abs(rec) - np.abs(target)) / np.mean(np.abs(target))
    if return_map:
        return map 
    else:
        return np.mean(map)


def phase_metric(rec, target, return_map=False):
    phase_map = np.abs(np.angle(np.conjugate(rec) * target))
    weights = np.abs(target)
    if return_map:
        return phase_map * weights / np.max(weights)
    else:
        return np.average(phase_map, weights=weights)


def phase_metric_torch(rec, target, return_map=False):
    phase_map = torch.abs(torch.angle(torch.conj(rec) * target))
    weights = torch.abs(target)
    if return_map:
        return phase_map * weights / torch.max(weights)
    else:
        return torch.sum(phase_map * weights) / torch.sum(weights)
    

def perp_loss(rec: torch.Tensor, target: torch.Tensor):
    assert rec.is_complex()
    assert target.is_complex()

    mag_input = torch.abs(rec)
    mag_target = torch.abs(target)
    cross = torch.abs(rec.real * target.imag - rec.imag * target.real)

    angle = torch.atan2(rec.imag, rec.real) - torch.atan2(target.imag, target.real)
    ploss = torch.abs(cross) / (mag_input + 1e-8)

    aligned_mask = (torch.cos(angle) < 0).bool()

    final_term = torch.zeros_like(ploss)
    final_term[aligned_mask] = mag_target[aligned_mask] + (mag_target[aligned_mask] - ploss[aligned_mask])
    final_term[~aligned_mask] = ploss[~aligned_mask]
    return (
    (final_term + torch.nn.functional.mse_loss(mag_input, mag_target)).mean())