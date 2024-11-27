"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *
from models.unet import Unet
import matplotlib.pyplot as plt
import numpy as np



class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, h, w, comp).permute(0, 3, 1, 2), b # shape (batch, coil, complex=2, height,  width)

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, comp, h, w = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, comp, h, w).permute(0, 1, 3, 4, 2).contiguous() # shape (batch, coil, height,  width, complex=2)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / rss(x)


    def get_acs_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # want central 24 x 24 square
        acs_mask = torch.zeros_like(mask)
        h_diff, w_diff = (mask.shape[-2] - 24) // 2, (mask.shape[-1] - 24) // 2
        acs_mask[:, :, h_diff:-h_diff, w_diff:-w_diff] = 1
        return acs_mask
    
    def get_acs_mask_circle(self, mask: torch.Tensor, radius=16) -> torch.Tensor:
        center_x, center_y = mask.shape[-2] // 2 - 0.5, mask.shape[-1] // 2 - 0.5

        # Create a grid of indices
        x, y = torch.arange(0, 218), torch.arange(0, 170)
        xv, yv = torch.meshgrid(x, y)

        # Calculate the distance of each point from the center
        distance_from_center = (xv - center_x) ** 2 + (yv - center_y) ** 2

        # Create a mask for the circle
        acs_mask = distance_from_center <= radius ** 2

        return acs_mask.to(device=mask.device)

    
    def compute_model_per_coil(self, data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of model `model_name` in `self.models` per coil.

        Parameters
        ----------
        model_name: str
            Model to run.
        data: torch.Tensor
            Multi-coil data of shape (batch, coil, complex=2, height, width).

        Returns
        -------
        output: torch.Tensor
            Computed output per coil.
        """
        output = []
        for idx in range(data.size(1)):
            subselected_data = data.select(1, idx)
            output.append(self.unet(subselected_data))

        return torch.stack(output, dim=1)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor) -> torch.Tensor: 
        '''
        mask : b, c, h, w. torch.bool
        masked_kspace : b, c, h, w. complex64.
        '''
        masked_kspace = self.get_acs_mask(mask) * masked_kspace

        images, batches = self.chans_to_batch_dim(torch.view_as_real(to_img(masked_kspace)))
        est_smap = self.unet(images)

        # estimate sensitivities
        #est_smap = self.compute_model_per_coil(images)
        est_smap = torch.view_as_complex(self.batch_chans_to_chan_dim(est_smap, batches).contiguous())

        return self.divide_root_sum_of_squares(est_smap)



class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        input_channels,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        style: str ='coil_combine',
        return_smap=False,
        rss_output = False
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.sens_net = SensitivityModel(
            sens_chans,
            sens_pools
        )

        if style == 'coil_combine':
            output_channels = 2
        else:
            output_channels = input_channels

        self.cascades = nn.ModuleList(
            [VarNetBlock(Unet(input_channels, output_channels, chans=chans, num_pool_layers=pools), style=style) for _ in range(num_cascades)]
        )
        self.return_smap = return_smap
        self.rss_output = rss_output

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        '''
        :masked_kspace: current kspace estimate (B, C, nrow, ncol) - complex64
        :mask: sampling mask (B, C, nrow, ncol) - int8
        '''
        sens_maps = self.sens_net(masked_kspace, mask)

        masked_kspace = to_re_imag(masked_kspace)
        kspace_pred = masked_kspace.clone()

        mask = mask.repeat(1, 2, 1, 1)
        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        if self.return_smap:
            return to_re_imag(coil_combine(to_img(to_complex(kspace_pred)), sens_maps, rss=self.rss_output)), sens_maps
        else:
            return to_re_imag(coil_combine(to_img(to_complex(kspace_pred)), sens_maps, rss=self.rss_output))


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module, style: str):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
                style: one of coil_combine or all_coils. Determine whether refinement is applied to
                combined or uncombined image. Note: kspace from dataloader is always uncombined
        """
        super().__init__()

        self.model = model
        self.style = style
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        '''
        :current_kspace: current kspace estimate (B, 2 * C, nrow, ncol) - float32
        :ref_kspace: original undersampled kspace (B, 2 * C, nrow, ncol) - float32
        :mask: sampling mask (B, C, nrow, ncol) - int8
        :sens_maps: coil sensitivity map (B, C, nrow, ncol) - complex64
        '''
        zero = torch.zeros(1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        if self.style == 'coil_combine':
            est_img = self.model(to_re_imag(coil_combine(to_img(to_complex(current_kspace)), sens_maps)))
            model_term = to_re_imag(to_kspace(expand(to_complex(est_img), sens_maps)))
        else:
            est_img = self.model(to_re_imag(to_img(to_complex(current_kspace))))
            model_term = to_re_imag(to_kspace(to_complex(est_img)))

        return current_kspace - soft_dc + model_term



def test(): 
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision.transforms.functional import to_tensor
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    # B, C, H, W
    coils = 4
    input_channels = coils * 2
    kspace = torch.complex(torch.randn(size=(5, coils, 218, 170)), torch.randn(size=(5, coils, 218, 170))).to(device)
    mask = torch.rand(size=(5, coils, 218, 170)) < 0.7
    #mask = np.load('undersampling_masks/218_170/R5_218x170.npy')
    #mask = np.load('undersampling_masks/218_170/vdpd_mask_R=4_v0.npy')
    #mask = np.repeat(mask[..., None], repeats=kspace.shape[1], axis=-1)
    #mask = np.repeat(mask[None, ...], repeats=5, axis=0)
    #mask = to_tensor(mask)
    mask = mask.to(device)

    model = VarNet(2, style='coil_combine', num_cascades=5).type(torch.float32).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    alt_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable params: {pytorch_total_params}")

    img_output = model(kspace, mask)
    print(kspace.shape, img_output.shape)

if __name__ == "__main__":
    test()