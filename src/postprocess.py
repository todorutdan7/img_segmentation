"""Clean a predicted label volume in 3D, after the model has run.

The model sees one slice at a time, so it can put liver or tumour where there is
no liver: a blob in the spleen, a speck in the stomach wall. Stacked into 3D
those are easy to spot - they are not connected to the liver.

    clean_volume(pred)        (z, y, x) labels 0/1/2 -> the same, cleaned

  1. liver and tumour together are the organ. Only its largest 3D connected
     piece is kept; every other piece becomes background. A tumour lies inside
     the liver, so a tumour that does not touch that piece cannot be one.
  2. optionally, tumour pieces smaller than min_tumor_voxels become liver.
     Off by default: small lesions are exactly what the model already misses.

No retraining - it runs on any model's output, at evaluation or in the viewer.
"""

import numpy as np
from scipy import ndimage

# 26-connectivity: diagonal neighbours count too, so an organ cut by thick
# slices or a thin bridge still counts as one piece
FULL = np.ones((3, 3, 3), bool)


def clean_volume(pred, min_tumor_voxels=0):
    """IN   pred - (z, y, x) uint8, 0 background / 1 liver / 2 tumour, the
                  slices of one patient in order
         min_tumor_voxels - tumour pieces smaller than this become liver
    OUT  a cleaned copy, same shape and dtype
    """
    out = pred.copy()
    organ = out > 0
    if organ.any():
        cc, n = ndimage.label(organ, structure=FULL)
        if n > 1:
            # sizes only over the organ's own voxels: a bincount of the whole
            # volume would copy it to int64 first (half a GB at 512)
            sizes = np.bincount(cc[organ])
            out[organ & (cc != int(np.argmax(sizes)))] = 0
    if min_tumor_voxels > 0:
        tumor = out == 2
        cc, n = ndimage.label(tumor, structure=FULL)
        if n:
            small = np.bincount(cc[tumor], minlength=n + 1) < min_tumor_voxels
            small[0] = False
            out[tumor & small[cc]] = 1
    return out
