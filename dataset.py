import h5py
import numpy as np
import pandas as pd
import random

from dataset_utils import *

import torch
from torch.utils.data import Dataset

from torchvision.transforms.functional import center_crop, to_tensor


class MRVolumeDataset(Dataset):
    """
    Dataset class for coil combine model when using 12 channel data. Here, the
    input to the model is coil combined, and only uncombined for the data
    consistency steps.

    12-channel kspace is already in re/im format, but smaps are not!

    Style is one of coil_combine or all_coils.
    """

    def __init__(
        self,
        kspace_filepaths: list,
        smap_filepaths: list,
        mask_filepaths: list,
        split: str,
        style: str,
        modulated=False,
        rss_output=False,
        sc_target=True,
        select_slice=None,
    ) -> None:
        super().__init__()
        self.kspace_filepaths = kspace_filepaths
        self.smap_filepaths = smap_filepaths
        self.mask_filepaths = mask_filepaths
        self.split = split
        self.style = style
        self.modulated = modulated
        self.rss_output = rss_output
        self.sc_target = sc_target

        if not select_slice:
            if split.lower() == "train":
                start, end = 55, 201  # end is inclusive
            else:
                start, end = 78, 178  # end is inclusive
        else:
            start, end = select_slice, select_slice

        # populate this dataframe so that for each volume filepath, we have
        # specified which slices we want to take from it
        self.len = 0
        self.metadata_temp = pd.DataFrame(
            columns=[
                "kspace file path",
                "kspace slice number",
                "smap file path",
                "smap slice number",
            ]
        )

        for index, kspace_filepath in enumerate(self.kspace_filepaths):
            smap_filepath = self.smap_filepaths[index]
            for i in range(start, end + 1):
                row = [kspace_filepath, i, smap_filepath, i - 55]
                self.metadata_temp.loc[len(self.metadata_temp)] = row
                self.len += 1

    def __len__(self):
        return int(self.len)

    def __getitem__(self, idx):

        # kspace_filepath, kspace_slice_num, smap_filepath, smap_slice_num = self.metadata_temp.iloc[idx]

        ## Deal with kspace ##
        kspace_filepath, kspace_slice_num, smap_filepath, smap_slice_num = (
            self.metadata_temp.iloc[idx]
        )
        with h5py.File(kspace_filepath, "r") as hf:
            kspace = hf["kspace"][kspace_slice_num]  # ky, kz, Nc x 2

        # to_tensor will move channels from end to front giving Nc x 2, ky, kz
        kspace = to_tensor(kspace)

        # modulate kspace
        if not self.modulated:
            kspace = modulate(kspace)

        # crop kspace
        if kspace.shape[-1] != 170:
            kspace = center_crop(kspace, (218, 170))

        # undersample
        mask = np.load(random.choice(self.mask_filepaths))

        # undersample kspace
        # remember to_tensor moves channels to front!
        kspace = to_complex(kspace)
        mask = np.repeat(mask[..., None], repeats=kspace.shape[0], axis=-1)
        mask = to_tensor(mask)
        us_kspace = kspace * mask

        # create images
        us_image = to_img(us_kspace)
        image = to_img(kspace)

        # scale by image values since this is an image domain problem
        # us_image, scale_factor = scale_by_re_im(us_image)  # scales by uncombined image
        # us_kspace /= scale_factor
        # image /= scale_factor
        # target_image /= scale_factor

        ## Load smaps ##
        with h5py.File(smap_filepath, "r") as hf:
            smap = hf["smap"][smap_slice_num]  # ky, kz, Nc
        smap = to_tensor(smap)

        if self.style == "coil_combine":
            # if we're coil combining the input, scale input based off of coil combined channels
            us_image = to_re_im(coil_combine(us_image, smap))
            us_image, scale_factor = scale_by_re_im(us_image)

        else:
            # if we're not coil combining, then scale input based off the channels
            us_image, scale_factor = scale_by_re_im(us_image)
            us_image = to_re_im(us_image)

        us_kspace /= scale_factor
        image /= scale_factor
        if self.sc_target:
            target_image = coil_combine(image, smap, rss=self.rss_output)

            target_image = to_re_im(
                target_image
            )  # note that imaginary component will be 0 if we applied RSS
        else:
            target_image = to_re_im(image)

        if self.split == "test":
            return us_image, target_image, us_kspace, smap, mask, kspace_filepath, kspace_slice_num
        else:
            return us_image, target_image, us_kspace, smap, mask


class GBMDataset(Dataset):
    """
    Dataset class for coil combine model when using 12 channel data. Here, the
    input to the model is coil combined, and only uncombined for the data
    consistency steps.

    12-channel kspace is already in re/im format, but smaps are not!

    Style is one of coil_combine or all_coils.
    """

    def __init__(
        self,
        kspace_filepaths: list,
        smap_filepaths: list,
        mask_filepaths: list,
        style: str,
        rss_output=False,
        sc_target=True,
    ) -> None:
        super().__init__()
        self.kspace_filepaths = kspace_filepaths
        self.smap_filepaths = smap_filepaths
        self.mask_filepaths = mask_filepaths
        self.rss_output = rss_output
        self.sc_target = sc_target
        self.style = style

    def __len__(self):
        return len(self.kspace_filepaths)

    def __getitem__(self, idx):

        kspace_filepath = self.kspace_filepaths[idx]
        smap_filepath = self.smap_filepaths[idx]

        # GBM data is in complex format to start
        kspace = np.load(kspace_filepath)
        # to_tensor will move channels from end to front giving Nc, ky, kz
        kspace = to_tensor(kspace)
        kspace = modulate(kspace)

        # undersample
        mask = np.load(random.choice(self.mask_filepaths))

        # undersample kspace
        # remember to_tensor moves channels to front!
        mask = np.repeat(mask[..., None], repeats=kspace.shape[0], axis=-1)
        mask = to_tensor(mask)
        us_kspace = kspace * mask

        # create images
        us_image = to_img(us_kspace)
        image = to_img(kspace)

        ## Load smaps ##
        smap = np.load(smap_filepath)
        smap = to_tensor(smap)

        if self.style == "coil_combine":
            # if we're coil combining the input, scale input based off of coil combined channels
            us_image = to_re_im(coil_combine(us_image, smap))
            us_image, scale_factor = scale_by_re_im(us_image)

        else:
            # if we're not coil combining, then scale input based off the channels
            us_image, scale_factor = scale_by_re_im(us_image)
            us_image = to_re_im(us_image)

        us_kspace /= scale_factor
        image /= scale_factor
        if self.sc_target:
            target_image = coil_combine(image, smap, rss=self.rss_output)
            target_image = to_re_im(
                target_image
            )  # note that imaginary component will be 0 if we applied RSS

        else:
            target_image = to_re_im(image)

        # GBM is always for testing
        return us_image, target_image, us_kspace, smap, mask, kspace_filepath
