"""
This model coil-combines input and only expands for data consistency steps.
"""
from models.model_blocks import *
from models.model_utils import *

import torch
from torch.nn import Sequential, Module



class CascadedModel(Module):
    def __init__(self, input_channels, blocks=5, block_depth=5, filters=110, residual=False, style='coil_combine') -> None:
        super().__init__()
        self.style = style
        # can iterate through a block or go through layers
        blocks_list = []

        if self.style == 'coil_combine':
            for i in range(blocks - 1):
                blocks_list.append(CascadeCoilCombineBlock(input_channels, block_depth, filters, residual=residual))
            blocks_list.append(CascadeCoilCombineBlock(input_channels, block_depth, filters, final=True, residual=residual))
        else:
            for i in range(blocks - 1):
                blocks_list.append(CascadeBlock(input_channels, block_depth, filters, residual=residual))
            blocks_list.append(CascadeBlock(input_channels, block_depth, filters, final=True, residual=residual))
        
        self.blocks = Sequential(*blocks_list)

    def forward(self, info_tuple):  # info tuple is img input, kspace, mask, smap
        """
        :input: zero-filled reconstruction (B, 2 * ncoil, H, W) - float32
        :kspace: raw undersampled k-space (B, 2 * ncoil, H, W) - float32
        :mask: sampling mask (B, ncoil, H, W) - int8
        :smap: coil sensitivity map (B, ncoil, H, W) - complex64
        """
        x, _, _, smap = self.blocks(info_tuple)

        if self.style == 'coil_combine':
            return x
        
        else:
            x = to_complex(x)
            sc_output = to_re_imag(coil_combine(x, smap))
            
            return sc_output
    