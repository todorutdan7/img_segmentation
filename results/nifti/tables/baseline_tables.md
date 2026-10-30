# NIfTI 512 baseline: U-Net and Attention U-Net

Trained on MSD Task03_Liver at 512x512 in Hounsfield units, RTX 2070, Oct 26-27 2026.
Runs: `runs/nifti/baseline/attention_unet_1793072064`, `runs/nifti/baseline/unet_1793049813`.

## Training setup

| Setting | Value |
| --- | --- |
| Models | U-Net (7,762,531 parameters) and Attention U-Net (7,894,783) |
| Architecture | 32 base filters, depth 4 (512 -> 32 x 32 at the bottleneck), batch norm, transposed-conv upsampling, no dropout |
| Input | 1 x 512 x 512, HU window [-17, 201] |
| Slices | 13,756 train / 3,155 validation / 2,252 test (only slices with liver) |
| Augmentation | random horizontal and vertical flips, 90° rotations |
| Loss | 0.5 x Dice + 0.5 x cross-entropy, no class weights |
| Optimiser | Adam, learning rate 0.001, weight decay 0.0001 |
| Schedule | 40 epochs, cosine decay of the learning rate to 0 |
| Batch | 8 slices, mixed precision (AMP) |
| Model kept | the epoch with the best validation foreground Dice |
| Hardware | RTX 2070 (8 GB), PyTorch 2.6, data on HDD, 4 loader workers |
| Training time | U-Net 6.2 h, Attention U-Net 7.3 h |

## Test results (20 patients, 2,252 slices)

| Model | Foreground Dice | Liver Dice | Tumour Dice (pooled) | Tumour Dice (per patient) | Tumour precision | Tumour recall | Best epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Attention U-Net | 0.816 | 0.916 | 0.715 | 0.367 | 0.714 | 0.717 | 23 |
| U-Net | 0.800 | 0.913 | 0.687 | 0.330 | 0.646 | 0.733 | 18 |

Pooled = Dice over all test pixels together; per patient = Dice for each patient, then averaged.

## Tumour Dice per test patient, by tumour size

| Test patient | Tumour (mL) | Lesions | Tumour Dice, Attention U-Net | Tumour Dice, U-Net |
| --- | --- | --- | --- | --- |
| 106 | 0 | 0 | 0.00 | 0.00 |
| 91 | 0 | 0 | 0.00 | 0.00 |
| 25 | 0.2 | 1 | 0.06 | 0.02 |
| 112 | 0.4 | 4 | 0.19 | 0.03 |
| 54 | 1.8 | 1 | 0.00 | 0.00 |
| 121 | 2.0 | 3 | 0.00 | 0.00 |
| 62 | 2.7 | 2 | 0.38 | 0.28 |
| 68 | 5.4 | 1 | 0.05 | 0.04 |
| 53 | 6.5 | 2 | 0.49 | 0.53 |
| 75 | 7.8 | 4 | 0.56 | 0.15 |
| 77 | 12.9 | 2 | 0.44 | 0.42 |
| 30 | 14.3 | 1 | 0.23 | 0.20 |
| 1 | 16.4 | 12 | 0.69 | 0.70 |
| 78 | 18.9 | 14 | 0.62 | 0.53 |
| 23 | 22.4 | 1 | 0.67 | 0.61 |
| 104 | 134.6 | 10 | 0.01 | 0.02 |
| 16 | 147.6 | 13 | 0.86 | 0.83 |
| 71 | 221.1 | 7 | 0.42 | 0.53 |
| 51 | 305.1 | 11 | 0.83 | 0.85 |
| 117 | 382.5 | 79 | 0.85 | 0.85 |

Patients 91 and 106 have no tumour; they score 0 because a few false tumour pixels are predicted.
