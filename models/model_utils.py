"""
Utilities and functions used in the models.
"""


import torch

def to_img(batch, dim1=-1, dim2=-2, **kwargs):
    """Convert kspace to image"""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(batch, dim=(dim1, dim2)), dim=(dim1, dim2), **kwargs),  
        dim=(dim1, dim2)
    )

def to_kspace(batch, dim1=-1, dim2=-2, **kwargs):
    """Convert image to kspace"""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(batch, dim=(dim1, dim2)), dim=(dim1, dim2), **kwargs), 
        dim=(dim1, dim2)
    )


def to_re_imag(batch): 
    """Converts complex data into two real channels"""
    b, c, h, w = batch.shape
    empty_batch = torch.empty((b, c * 2, h , w), device=batch.device)
    try:
        re_img, im_img = torch.real(batch), torch.imag(batch)
    except:  # if it's a non-complex input
        re_img, im_img = batch, torch.zeros_like(batch)
    empty_batch[:, ::2, :, :] = re_img
    empty_batch[:, 1::2, :, :] = im_img
    return empty_batch #.to('cuda:0')


def to_complex(batch):
    """Converts two real channels into complex data"""
    batch = batch[:, ::2, :, :] + 1j * batch[:, 1::2, :, :]
    return batch


def expand(image:torch.tensor, smap:torch.tensor):
    """Do complex coil combination on image of size Nc, ky, kz"""
    return image * smap


def coil_combine(image:torch.tensor, smap:torch.tensor, rss=False):
    """Do complex coil combination on image of size batch, Nc, ky, kz"""
    if rss:
        return torch.sqrt(torch.sum(torch.square(torch.abs(image)), dim=1, keepdim=True))
    else:
        return torch.sum(image * torch.conj(smap), dim=1, keepdim=True)


def rss(image: torch.tensor):
    """ Do rss on complex image of size B, C, H, W"""
    return torch.sqrt(torch.sum(torch.abs(image) ** 2, dim=1, keepdim=True))