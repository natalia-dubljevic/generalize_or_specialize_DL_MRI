import matplotlib.pyplot as plt
import numpy as np
from sigpy.mri import poisson

# Want R=4 and 8, but recall that in CC dataset 15% is already gone!
# Along nz, we don't sample the last 15%, which will be reflected in the 
# udnersampling masks.
ny, nz = 218, 170
target_R = 8
R = 7.5  # use 3.7 to get effective 4 for 218 x 170 image, and 7.5 to get effective R=8

for i in range(50):
    cutoff = int(np.ceil(nz * 0.85))  # Last 15% along nz is not sampled
    mask = poisson((ny, nz), R, calib=(24, 24), crop_corner=False, dtype=np.int32, max_attempts=20, seed=i)
    mask[:, cutoff:] = 0
    print(f'Effective R: {ny * nz / np.sum(mask)}')
    # plt.imshow(mask)
    # plt.savefig('test.png')
    np.save(f'undersampling_masks/218_{nz}/vdpd_mask_R={target_R}_v{i}.npy', mask)
    print(i)