"""
Classes for the different blocks used in models.
"""

import torch

from torch import nn
from models.model_utils import *


class DataConsistency(nn.Module):
    """Makes sure known k-space points are taken into consideration"""
    def __init__(self, complex=False):
        super().__init__()
        self.lamda = nn.Parameter(torch.ones(1))
        self.complex = complex

    def forward(self, img, og_kspace, mask, **kwargs):
        if self.complex:
            kspace_pred = to_kspace(img, **kwargs)
        else:
            kspace_pred = to_kspace(to_complex(img), **kwargs)
        diff = kspace_pred - og_kspace
        zeros = torch.zeros_like(diff, dtype=torch.complex64)
        masked = torch.where(mask, diff, zeros)
        masked *= self.lamda

        updated = kspace_pred - masked

        if self.complex:
            return to_img(updated, **kwargs)
        else:
            return to_re_imag(to_img(updated, **kwargs))


class CascadeBlock(nn.Module):
    """
    Assumes coils stay uncombined throughout
    """
    def __init__(self, input_channels, block_depth, filters, final=False, residual=False) -> None:
        super().__init__()

        self.residual = residual
        self.final = final

        layers = []
        layers.append(nn.Conv2d(input_channels, filters, kernel_size=3, padding='same'))
        layers.append(nn.LeakyReLU())

        for i in range(block_depth - 2):
            layers.append(nn.Conv2d(filters, filters, kernel_size=3, padding='same'))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Conv2d(filters, input_channels, kernel_size=3, padding='same'))

        if not self.final:
            layers.append(nn.LeakyReLU())

        self.layers = nn.Sequential(*layers)
        self.dc = DataConsistency()

    def forward(self, info_tuple) -> torch.Tensor:
        x0, og_kspace, mask, smap = info_tuple
        x = self.layers(x0)
        # do dc layer, unless it's the last one...give the model some freedom
        #if not self.final:
        x = self.dc(x, og_kspace, mask)

        if self.residual:
            x += x0

        return (x, og_kspace, mask, smap)
    

class CascadeCoilCombineBlock(nn.Module):
    """
    Assumes coils are combined and must be uncombined for DC
    """
    def __init__(self, input_channels, block_depth, filters, final=False, residual=False) -> None:
        super().__init__()
        
        self.residual = residual
        layers = []
        layers.append(nn.Conv2d(input_channels, filters, kernel_size=3, padding='same'))
        layers.append(nn.LeakyReLU())

        for i in range(block_depth - 2):
            layers.append(nn.Conv2d(filters, filters, kernel_size=3, padding='same'))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Conv2d(filters, input_channels, kernel_size=3, padding='same'))

        if not final:
            layers.append(nn.LeakyReLU())

        self.layers = nn.Sequential(*layers)
        self.dc = DataConsistency(complex=True)

    def forward(self, info_tuple) -> torch.Tensor:
        x0, og_kspace, mask, smap = info_tuple
        x = self.layers(x0)
        # do dc layer
        x = expand(to_complex(x), smap)
        x = self.dc(x, og_kspace, mask)
        x = to_re_imag(coil_combine(x, smap))

        if self.residual:
            x += x0

        return (x, og_kspace, mask, smap)