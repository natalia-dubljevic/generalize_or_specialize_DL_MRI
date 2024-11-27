from dataset import MRVolumeDataset, GBMDataset
import matplotlib.pyplot as plt
from metrics import *
from models.DC import CascadedModel
from models.model_utils import to_complex
from models.MoDL import MoDL
from models.E2Evarnet import VarNet
from training_utils import *

from torch.utils.data import DataLoader

from datetime import datetime
import glob
import numpy as np
import random
import torch
import sys
from pathlib import Path
import pandas as pd
from dataset_utils import max_min_scale

dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# a bunch of imports
version, style, model_type, input_channels = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
)
blocks, block_depth, filters = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
kspace_paths, smap_paths = sys.argv[8], sys.argv[9]
modulated, residual = int(sys.argv[10]), int(sys.argv[11])
R, rss = int(sys.argv[12]), int(sys.argv[13])
sc_target, num_workers = (
    int(sys.argv[14]),
    int(sys.argv[15]),
)
GBM = int(sys.argv[16])

# initiate some random seeds and check cuda
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device, flush=True)

# paths to data and masks
kspace_paths = sorted(glob.glob(kspace_paths))
smap_paths = sorted(glob.glob(smap_paths))

mask_paths = glob.glob(
    f"/home/natalia.dubljevic/compare_DL_MR_recon/undersampling_masks/218_170/vdpd_mask_R={R}_*.npy"
)

# Style is one of coil_combine, all_coils.
if GBM:
    data = GBMDataset(
        kspace_paths,
        smap_paths,
        mask_paths,
        style,
        rss_output=rss,
        sc_target=sc_target,
    )

else:  # healthy calgary-campinas data
    data = MRVolumeDataset(
        kspace_paths,
        smap_paths,
        mask_paths,
        "test",
        style,
        modulated=modulated,
        rss_output=rss,
        sc_target=sc_target,
    )


data_loader = DataLoader(
    data,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=num_workers,
    batch_size=1,
)

# define model
if model_type == "DC":
    model = CascadedModel(
        input_channels,
        blocks=blocks,
        filters=filters,
        block_depth=block_depth,
        residual=residual,
        style=style
    ).type(torch.float32)


elif model_type == "MoDL":
    model = MoDL(
        input_channels=input_channels,
        n_filters=filters,
        style=style,
        k_iters=blocks,
        n_layers=block_depth,
        sc_target=sc_target,
    ).type(torch.float32)

else:
    model = VarNet(
        input_channels,
        num_cascades=blocks,
        chans=filters,
        style=style,
        pools=block_depth,
        rss_output=rss,
    ).type(torch.float32)


model.to(device)
model_checkpoint = torch.load(f"model_weights/{model_type}_{style}_V_{version}.pt")
model.load_state_dict(model_checkpoint["model_state_dict"])
model.to(device)
model.eval()

# set up lists to collect metrics
ssims = []
psnrs = []
apds = []
img_ids = []
slice_nums = []

fig_img = plt.figure(figsize=(24, 18))
fig_phase = plt.figure(figsize=(24, 18))
for i, data in enumerate(data_loader):
    print(i, flush=True)
    # load data
    us_img, img_label = data[0].to(device, dtype=torch.float32), data[1].to(
        device, dtype=torch.float32
    )
    us_kspace = data[2].to(device, dtype=torch.complex64)
    smap, mask = data[3].to(device, dtype=torch.complex64), data[4].to(
        device, dtype=torch.bool
    )

    filepath = data[5][0]
    img_id = Path(filepath).stem
    img_ids.append(img_id)

    if GBM:
        slice = int(img_id.split("_")[-1][1:])
        slice_nums.append(slice)
    else:
        slice = int(data[6][0])
        slice_nums.append(slice)

    if model_type == "DC":
        pred_img = model((us_img, us_kspace, mask, smap))
    elif model_type == "MoDL":
        pred_img = model(us_img, smap, mask)
    else:  # e2evarnet
        pred_img = model(us_kspace, mask)

    # recall outputs will have real/imaginary channels even if RSS output!
    # move to cpu, make complex, and convert to numpy array
    pred_img = to_complex(pred_img.detach().cpu()).numpy().squeeze()
    img_label = to_complex(img_label.detach().cpu()).numpy().squeeze()

    # scale max to 1
    pred_img_abs = max_min_scale(np.abs(pred_img))
    img_label_abs = max_min_scale(np.abs(img_label))

    # generate metrics
    ssim = SSIM(pred_img_abs, img_label_abs)
    psnr = pSNR(pred_img_abs, img_label_abs)
    apd = phase_metric(pred_img, img_label)

    ssims.append(ssim)
    psnrs.append(psnr)
    apds.append(apd)

    if i in np.arange(0, 10):
        # do images
        ax = fig_img.add_subplot(5, 4, (i + 1) * 2 - 1)
        ax.imshow(img_label_abs)
        ax.set_title(f"SSIM: {ssim:.3f}")
        ax = fig_img.add_subplot(5, 4, (i + 1) * 2)
        ax.imshow(pred_img_abs)
        ax.set_title(f"pSNR: {psnr:.1f}")
        fig_img.savefig(
            f"/home/natalia.dubljevic/compare_DL_MR_recon/results/images/{model_type}_{style}_V_{version}_R={R}.png", bbox_inches='tight'
        )

        ax = fig_phase.add_subplot(5, 4, (i + 1) * 2 - 1)
        ax.imshow(np.angle(img_label))
        ax.set_title(f"APD: {apd:.3f}")
        ax = fig_phase.add_subplot(5, 4, (i + 1) * 2)
        ax.imshow(np.angle(pred_img))
        #plt.colorbar(ax=ax)
        fig_phase.savefig(
            f"/home/natalia.dubljevic/compare_DL_MR_recon/results/images/{model_type}_{style}_V_{version}_R={R}_phase.png", bbox_inches='tight'
        )

results = pd.DataFrame(
    {
        "img_id": img_ids,
        "slice_num": slice_nums,
        "R_factor": R,
        "ssim": ssims,
        "psnr": psnrs,
        "apd": apds,
    }
)
if GBM:
    dataset = "GBM"
else:
    dataset = "CC"
results.to_csv(
    f"results/{model_type}_{style}_V_{version}_R={R}_{dataset}.csv",
    index=False,
)