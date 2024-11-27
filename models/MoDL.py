import torch
import torch.nn as nn
from models.model_utils import *

"""
adapted from https://github.com/bo-10000/MoDL_PyTorch
"""


# CNN denoiser ======================
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding="same"),
        #nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class cnn_denoiser(nn.Module):
    def __init__(self, n_layers, n_filters, input_channels):
        super().__init__()
        layers = []
        layers += conv_block(input_channels, n_filters)

        for _ in range(n_layers - 2):
            layers += conv_block(n_filters, n_filters)

        layers += nn.Sequential(
            nn.Conv2d(n_filters, input_channels, 3, padding="same"),
            #nn.BatchNorm2d(input_channels),
        )

        self.nw = nn.Sequential(*layers)

    def forward(self, x):
        idt = x  # (2, nrow, ncol)
        dw = self.nw(x) + idt  # (2, nrow, ncol)
        return dw


# CG algorithm ======================
class myAtA(nn.Module):
    """
    performs DC step
    """

    def __init__(self, csm, mask, lam, style):
        super(myAtA, self).__init__()
        self.csm = csm  # complex (B x ncoil x nrow x ncol). sensitivity maps
        self.mask = mask  # complex (B x nrow x ncol)
        self.lam = lam
        self.style = style

    def forward(self, im):  # step for batch image
        """
        :im: complex image (B x nrow x nrol)
        """
        im_coil = self.csm * im  # split coil images (B x ncoil x nrow x ncol)

        k_full = to_kspace(im_coil, norm="ortho")  # convert into k-space
        k_u = k_full * self.mask  # undersampling
        im_u_coil = to_img(k_u, norm="ortho")  # convert into image domain

        im_u = coil_combine(im_u_coil, self.csm)  # coil combine (B x nrow x ncol)

        return im_u + self.lam * im


def myCG(AtA, rhs):
    """
    performs CG algorithm
    :AtA: a class object that contains csm, mask and lambda and operates forward model

    CG operates on coil-combined image even if we have separate channels for convolutions.
    See pruessmann 2001 paper
    """
    rhs = to_complex(rhs)  # nrow, ncol
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = torch.sum(
        r.conj() * r, dim=(1, 2, 3), keepdim=True
    ).real  # .real  this returns a single value for whole batch??
    while i < 10 and rTr.abs().max() > 1e-10:
        Ap = AtA(p)  # complex valued, gives a val per batch item
        alpha = rTr / torch.sum(p.conj() * Ap, dim=(1, 2, 3), keepdim=True).real
        alpha = alpha  # real valued
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(r.conj() * r, dim=(1, 2, 3), keepdim=True).real  # .real
        beta = rTrNew / rTr
        beta = beta
        p = r + beta * p  # complex valued
        i += 1
        rTr = rTrNew
    return to_re_imag(x)


class data_consistency(nn.Module):
    """
    If coil data is multi-channel (not combined), combine it here-- it makes the
    rest more straightforward. CG SENSE requires coil combined input...
    """

    def __init__(self, style):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)
        self.style = style

    def forward(self, z_k, x0, csm, mask):
        if self.style != "coil_combine":
            z_k = to_re_imag(coil_combine(to_complex(z_k), csm))

        rhs = x0 + self.lam * z_k  # (2, nrow, ncol)
        AtA = myAtA(csm, mask, self.lam, self.style)
        rec = myCG(AtA, rhs)  # output of this will be coil-combined

        if self.style != "coil_combine":
            rec = to_re_imag(expand(to_complex(rec), csm))

        return rec


# model =======================
class MoDL(nn.Module):
    def __init__(
        self,
        n_layers,
        k_iters,
        n_filters=64,
        input_channels=2,
        style="coil_combine",
        sc_target=True,
    ):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        :n_filters: number of filters
        :input_size: number of input channels
        :style: either coil_combine as in original, or all coils ie) not combined input
        """
        super().__init__()
        self.style = style
        self.k_iters = k_iters
        self.dw = cnn_denoiser(n_layers, n_filters, input_channels)
        self.dc = data_consistency(self.style)
        self.sc_target = sc_target

    def forward(self, x0, csm, mask):
        """
        :x0: zero-filled reconstruction (B, 2 or 2*ncoil, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """

        x_k = x0.clone()
        if self.style != "coil_combine":
            x0 = to_re_imag(coil_combine(to_complex(x0), csm))

        for k in range(self.k_iters):
            # dw
            z_k = self.dw(x_k)  # (2, nrow, ncol)
            # dc
            x_k = self.dc(z_k, x0, csm, mask)  # (2, nrow, ncol)

        if self.style != "coil_combine" and self.sc_target:
            x_k = to_re_imag(coil_combine(to_complex(x_k), csm))

        return x_k


def test():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    # B, C, H, W
    input_img = torch.randn(size=(5, 8, 256, 256), dtype=torch.float32).to(
        device
    )  # size 1, 2, 256, 256
    ref_kspace = torch.complex(
        torch.randn(size=(5, 4, 256, 256)), torch.randn(size=(5, 4, 256, 256))
    ).to(device)
    csm = torch.complex(
        torch.randn(size=(5, 4, 256, 256)), torch.randn(size=(5, 4, 256, 256))
    ).to(device)
    mask = torch.rand(size=(5, 4, 256, 256)) < 0.7
    mask = mask.to(device)

    model = (
        MoDL(input_channels=8, n_layers=4, k_iters=10, style="all_coils")
        .type(torch.float32)
        .to(device)
    )
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    alt_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable params: {pytorch_total_params}")

    input_tuple = (input_img, csm, mask)
    img_output = model(input_img, csm, mask)
    print(input_img.shape, img_output.shape)


if __name__ == "__main__":
    test()
