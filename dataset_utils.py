import torch
import numpy as np

def modulate(kspace: torch.tensor):
    """Modulate along ky, kz. Assumes tensor is of size Nc, ky, kz"""
    ones = torch.ones((kspace.shape))
    ones[:, 1::2, :] *= -1
    ones[:, :, 1::2] *= -1
    kspace = kspace * ones
    
    return kspace


def scale_by_re_im(image: torch.tensor):
    """Scale complex image by max real or imaginary component"""
    if torch.is_complex(image):
        scale_factor = torch.max(torch.abs(torch.view_as_real(image)))
    else:
        scale_factor = torch.max(torch.abs(image))
    return image / scale_factor, scale_factor


def to_img(kspace:torch.tensor, dim1=-1, dim2=-2):
    """Convert kspace to image"""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(dim1, dim2)), dim=(dim1, dim2)), 
        dim=(dim1, dim2)
    )

def coil_combine(image:torch.tensor, smap:torch.tensor, rss=False):
    """Do complex coil combination on image of size Nc, ky, kz"""
    if rss:
        return torch.sqrt(torch.sum(torch.square(np.abs(image)), dim=0, keepdim=True))
    else:
        return torch.sum(image * torch.conj(smap), dim=0, keepdim=True)


def max_min_scale(image):
    return (image - image.min()) / (image.max() - image.min())


def to_re_im(kspace:torch.tensor):
    """Convert complex tensor to separate real/imaginary channels"""
    nc, ky, kz = kspace.shape
    re_im_batch = torch.empty((nc * 2, ky, kz))
    try:  # if data is complex
        re_img, im_img = torch.real(kspace), torch.imag(kspace)
    except:  # if data is already real-valued
        re_img, im_img = kspace, torch.zeros_like(kspace)
    re_im_batch[::2, :, :] = re_img
    re_im_batch[1::2, :, :] = im_img

    return re_im_batch


def to_complex(batch):
    """Converts two real channels into complex data"""
    batch = batch[::2, ...] + 1j * batch[1::2, ...]
    return batch