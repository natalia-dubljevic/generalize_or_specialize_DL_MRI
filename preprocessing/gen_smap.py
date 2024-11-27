import bart
import h5py
import numpy as np
from scipy import interpolate
import glob
import sys
import os

"""
Generate sesntivity maps for training volumes 
(55 <= slice # <= 201)

For validation and testing, it's 78 <= slice # < 178

Coil sensitvity extension adapted from https://github.com/nalinimsingh/neuroMoCo
"""


def extend_float_maps(fmap, mask, bg, x, y):
    mask_x, mask_y = np.where(mask)

    z = [fmap[mask_x[i], mask_y[i]] for i in range(mask_x.shape[0])]

    tck = interpolate.bisplrep(
        mask_x, mask_y, z, xb=0, xe=mask.shape[0], yb=0, ye=mask.shape[1], kx=3, ky=3
    )
    bspl = interpolate.bisplev(x, y, tck)

    return bspl


def extend_sens_maps(maps):
    ext_maps = np.zeros(maps.shape, dtype=np.complex64)

    for coil in range(maps.shape[2]):
        mag_map = np.abs(maps[..., coil])
        re_map = np.real(maps[..., coil])
        im_map = np.imag(maps[..., coil])

        mask = mag_map != 0
        bg = mag_map == 0

        x = np.arange(maps.shape[0])
        y = np.arange(maps.shape[1])

        re_bspl = extend_float_maps(re_map, mask, bg, x, y)
        im_bspl = extend_float_maps(im_map, mask, bg, x, y)

        bspl = re_bspl + 1j * im_bspl

        ext_maps[..., coil] = bspl

    norm = np.expand_dims(np.sqrt(np.sum(np.square(np.abs(ext_maps)), axis=2)), -1)
    ext_maps = np.divide(ext_maps, norm, out=np.zeros_like(ext_maps), where=norm != 0)

    return ext_maps

# Split: Should be one of Train, Val, or Test
# Channels: 12 for this experiment
# Read_path: Path to folder which contains raw h5 files
# Save_path: Path to folder where you want to save processed sensitivity maps
split, channels, read_path, save_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] 

filenames = glob.glob(os.path.join(read_path, f"*.h5"))

for filepath in filenames:
    kspace = h5py.File(filepath)["kspace"][:]

    # crop to 218 x 170 where applicable
    if kspace.shape[2] == 170:
        pass
    else:
        diff = int((kspace.shape[2] - 170) / 2)  # difference per side
        kspace = kspace[:, :, diff:-diff, :]
    # convert to complex
    kspace = kspace[..., ::2] + kspace[..., 1::2] * 1j

    # modulate to take out RF chopping
    ones = np.ones((kspace.shape))
    ones[:, 1::2, :, :] *= -1
    ones[:, :, 1::2, :] *= -1
    kspace = kspace * ones

    filename = filepath.split("/")[-1][0:-3]

    # generate espirit smaps on a slice by slice basis (doesn't work well in 3D)
    smap_stack = []
    for i in np.arange(55, 202):
        kslice = kspace[[i], ...]
        # generate sensitivity maps
        smaps = bart.bart(1, "ecalib -m 1", kslice)
        # extend them to cover the entire field of view
        ext_smaps = extend_sens_maps(np.squeeze(smaps))
        smap_stack.append(ext_smaps)

    # stack them such that the final shape is smap slices, y, z, channels
    vol_smaps = np.stack(smap_stack, axis=0).astype(np.complex64)

    h5_file = h5py.File(
        os.path.join(save_path, f"{channels}-channel/{split}/smaps/{filename}.h5"),
        "w",
    )
    h5_file.create_dataset("smap", data=vol_smaps)
    h5_file.close()

    print(f"Done {filename}", flush=True)
