from dataset import MRVolumeDataset
from metrics import *
from models.DC import CascadedModel
from models.model_utils import to_complex
from models.MoDL import MoDL
from models.E2Evarnet import VarNet
from training_utils import *

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader

from datetime import datetime
import glob
import numpy as np
import random
import torch
import sys
import wandb

dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# a bunch of imports
version, style, model_type, input_channels = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
)
blocks, block_depth, filters = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
epochs, lr, batch_size = int(sys.argv[8]), float(sys.argv[9]), int(sys.argv[10])
stopper_patience, plateau_patience = int(sys.argv[11]), int(sys.argv[12])
project, note = sys.argv[13], sys.argv[14]
train_kspace_paths, val_kspace_paths = sys.argv[15], sys.argv[16]
train_smap_paths, val_smap_paths = sys.argv[17], sys.argv[18]
modulated, residual = int(sys.argv[19]), int(sys.argv[20])
# schduler stuff
milestones, gamma, rss = sys.argv[21], float(sys.argv[22]), int(sys.argv[23])
num_workers, loss_fn_type = int(sys.argv[24]), sys.argv[25]
sc_target, resume, run_id = int(sys.argv[26]), int(sys.argv[27]), sys.argv[28]
init_model = sys.argv[29]

if not resume:
    run_id = wandb.util.generate_id()

config = {
    "version": version,
    "style": style,
    "model_type": model_type,
    "input_channels": input_channels,
    "epochs": epochs,
    "blocks": blocks,
    "block_depth": block_depth,
    "filters": filters,
    "batch_size": batch_size,
    "learning_rate": lr,
    "reduce_lr_patience": plateau_patience,
    "early_stopper_patience": stopper_patience,
    "date/time": dt_string,
    "run_id": run_id,
    "version": version,
    "residual": residual,
    "milestones": milestones,
    "gamma": gamma,
    "rss": rss,
    'loss type': loss_fn_type
}
#run_name = f"{model_type}_{style}_{version}"
run_name = f"{model_type}_{style}"

if resume:
    run = wandb.init(
        project=project, id=run_id, name=run_name, config=config, notes=note, resume="allow"
    )  # resume is True when resuming
    print(f'Resuming run {run_id}')
else:
    run = wandb.init(
        project=project, id=run_id, name=run_name, config=config, notes=note)
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

train_kspace_paths = sorted(glob.glob(train_kspace_paths))
val_kspace_paths = sorted(glob.glob(val_kspace_paths))
train_smap_paths = sorted(glob.glob(train_smap_paths))
val_smap_paths = sorted(glob.glob(val_smap_paths))

masks_paths_R_4 = glob.glob(
    r"/home/natalia.dubljevic/compare_DL_MR_recon/undersampling_masks/218_170/vdpd_mask_R=4_*.npy"
)
masks_paths_R_8 = glob.glob(
    r"/home/natalia.dubljevic/compare_DL_MR_recon/undersampling_masks/218_170/vdpd_mask_R=8_*.npy"
)

mask_paths = masks_paths_R_4 + masks_paths_R_8

# Style is one of coil_combine or all_coils.
train_data = MRVolumeDataset(
    train_kspace_paths,
    train_smap_paths,
    mask_paths,
    "train",
    style,
    modulated=modulated,
    rss_output=rss,
    sc_target=sc_target
)

val_data = MRVolumeDataset(
    val_kspace_paths,
    val_smap_paths,
    mask_paths,
    "val",
    style,
    modulated=modulated,
    rss_output=rss,
    sc_target=sc_target
)

# create dataloaders
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=num_workers,
)
valid_loader = DataLoader(
    val_data,
    batch_size=1,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=num_workers,
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
        sc_target=sc_target
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
model_save_path = f"model_weights/{model_type}_{style}_V_{version}.pt"

# define hyperparmaters
if loss_fn_type.lower() in ('apd_ssim', 'apd_mse'):
    weight_balance = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
    weight_balance.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + [weight_balance], lr=lr)
else:
    weight_balance = None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


if "_" in milestones:
    milestones = milestones.split("_")
    milestones = [int(m) for m in milestones]  # convert from string to itneger
else:
    milestones = [int(milestones)]

if resume:
    model, optimizer, last_epoch, prev_best_loss = load_checkpoint(model_save_path, model, optimizer)
    best_loss = prev_best_loss
    model_save_path = f"model_weights/{model_type}_{style}_V_{version}_resumed.pt"
    early_stopper = FracEarlyStopper(patience=stopper_patience, prev_best_loss=prev_best_loss)
    milestones = [m - last_epoch for m in milestones if m - last_epoch > 0]

    start_epoch, end_epoch = last_epoch + 1, last_epoch + epochs

else:
    start_epoch, end_epoch = 0, epochs

if init_model != 'none':
    model, _, _, _ = load_checkpoint(init_model, model, optimizer)
    print(f'Initializing model with weights from {init_model}')

print("Milestones")
print(milestones)

step_scheduler = MultiStepLR(optimizer, milestones, gamma=gamma)
threshold = 0.001
reduce_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=plateau_patience, threshold=threshold, threshold_mode='rel')

early_stopper = FracEarlyStopper(patience=stopper_patience, threshold=threshold)

best_loss = 1e20
### TRAIN LOOP ###
print(f"Started training model version {version}", flush=True)
for epoch in range(start_epoch, end_epoch):
    train_loss = 0.0
    train_ssim = 0
    for i, data in enumerate(train_loader, 0):
        us_img, img_label = data[0].to(device, dtype=torch.float32), data[1].to(
            device, dtype=torch.float32
        )
        us_kspace = data[2].to(device, dtype=torch.complex64)
        smap, mask = data[3].to(device, dtype=torch.complex64), data[4].to(
            device, dtype=torch.bool
        )

        optimizer.zero_grad()
        if model_type == "DC":
            output_img = model((us_img, us_kspace, mask, smap))
        elif model_type == "MoDL":

            output_img = model(us_img, smap, mask)
        else:  # e2evarnet
            output_img = model(us_kspace, mask)

        loss, mean_ssim = loss_criterion(
            output_img, img_label, loss_fn_type, return_ssim=True, 
            sc_target=sc_target, smap=smap, weight_balance=weight_balance
        )
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        train_ssim += mean_ssim.item()

    train_loss /= i + 1
    train_ssim /= i + 1
    print(
        f"{epoch + 1},  train loss: {train_loss:.6f}, train ssim: {train_ssim:.6f}",
        flush=True,
    )

    val_loss = 0
    val_ssim = 0
    ### VALIDATION LOOP ###
    with torch.no_grad():
        preds = []
        labels = []
        ssim_per_R = {4: [], 8: []}
        # smaps = []
        for i, data in enumerate(valid_loader, 0):
            us_img, img_label = data[0].to(device, dtype=torch.float32), data[1].to(
                device, dtype=torch.float32
            )
            us_kspace = data[2].to(device, dtype=torch.complex64)
            smap, mask = data[3].to(device, dtype=torch.complex64), data[4].to(
                device, dtype=torch.bool
            )

            if model_type == "DC":
                output_img = model((us_img, us_kspace, mask, smap))
            elif model_type == "MoDL":
                output_img = model(us_img, smap, mask)
            else:  # e2evarnet
                output_img = model(us_kspace, mask)

            loss, mean_ssim = loss_criterion(
                output_img, img_label, loss_fn_type, return_ssim=True, 
                sc_target=sc_target, smap=smap, weight_balance=weight_balance
            )

            val_loss += loss.item()
            val_ssim += mean_ssim.item()

            R = int(torch.round(1 / (torch.sum(mask) / torch.numel(mask))))
            ssim_per_R[R].append(mean_ssim.item())

            if i in range(4):
                if not sc_target:
                    img_pred = torch.abs(coil_combine(to_complex(output_img), smap))
                    img_pred = np.squeeze(img_pred.detach().cpu().numpy())
                else:
                    img_pred = output_img.detach().cpu().numpy()
                    img_pred = np.abs(img_pred[0, 0, :, :] + 1j * img_pred[0, 1, :, :])

                if not sc_target:
                    img_label = torch.abs(coil_combine(to_complex(img_label), smap))
                    img_label = np.squeeze(img_label.detach().cpu().numpy())
                else:
                    img_label = img_label.detach().cpu().numpy()
                    img_label = np.abs(img_label[0, 0, :, :] + 1j * img_label[0, 1, :, :])

                img_pred = wandb_scale_img(img_pred)
                img_label = wandb_scale_img(img_label)

                ssim = SSIM(img_pred, img_label)
                caption = f"R={R:.1f}, SSIM: {ssim:.3f}"

                preds.append(wandb.Image(img_pred, caption=caption))
                labels.append(wandb.Image(img_label, caption=caption))

        val_loss /= i + 1
        val_ssim /= i + 1
        mean_ssim_R_4 = np.mean(ssim_per_R[4])
        mean_ssim_R_8 = np.mean(ssim_per_R[8])
        print(f"val loss: {val_loss:.6f}, val ssim: {val_ssim:.6f}", flush=True)
        if weight_balance:
            print(f"weight balance: {weight_balance.item()}")
        # scheduler.step(val_loss)
        current_lr = step_scheduler.get_last_lr()
        reduce_scheduler.step(val_loss)
        step_scheduler.step()

        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_ssim": train_ssim,
                "val_ssim": val_ssim,
                "val_ssim_R_4": mean_ssim_R_4,
                "val_ssim_R_8": mean_ssim_R_8,
                "current_lr": current_lr[0],
                "pred": preds,
                "pred_label": labels,
            },
            step=epoch + 1,
        )

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            print("Saving model", flush=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                model_save_path,
            )

    if early_stopper.early_stop(val_loss):
        nepochs = epoch + 1
        break

print("Finished Training! :D", flush=True)
