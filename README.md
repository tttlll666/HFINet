Our code will be released upon acceptance of the paper.

# HFINet Experimental Setup and Dataset Splits

We evaluate HFINet for remote sensing image semantic segmentation on three datasets: LoveDA, ISPRS Potsdam, and ISPRS Vaihingen. This document describes the experimental environment, dataset splits, and training and testing settings.

## Experimental Environment

| Component              | Configuration                                          |
| ---------------------- | ------------------------------------------------------ |
| Operating system       | Ubuntu 20.04                                           |
| CPU                    | Intel Xeon Platinum 8352V, 16 vCPUs                    |
| GPU                    | One vGPU with 32 GB of memory                          |
| Python                 | 3.8                                                    |
| PyTorch                | 2.0.0                                                  |
| CUDA                   | 11.8                                                   |
| Numerical precision    | FP32                                                   |
| Encoder initialization | ResNet-18 weights from SWSL classification pretraining |

## Datasets

### Dataset Overview

| Dataset         | Input bands | Spatial resolution | Original image size | Number of classes |
| --------------- | ----------- | ------------------ | ------------------- | ----------------: |
| LoveDA          | RGB         | 0.3 m              | 1024 × 1024         |                 7 |
| ISPRS Potsdam   | IRRG        | 5 cm               | 6000 × 6000         |                 6 |
| ISPRS Vaihingen | IRRG        | 9 cm               | Variable            |                 6 |

IRRG denotes the near-infrared, red, and green bands. LoveDA contains seven classes: background, building, road, water, barren, forest, and agriculture. Potsdam and Vaihingen contain six classes: impervious surface, building, low vegetation, tree, car, and clutter.

### Split Summary

| Dataset         | Training images | Validation images | Test images |
| --------------- | --------------: | ----------------: | ----------: |
| LoveDA          |            2522 |              1669 |        1796 |
| ISPRS Potsdam   |              22 |                 1 |          14 |
| ISPRS Vaihingen |              15 |                 1 |          17 |

These counts refer to original images before cropping and augmentation. The training set is used for parameter optimization, the validation set for model selection, and the test set for final evaluation. Within each dataset, the three splits contain mutually exclusive original images.

### LoveDA

We use the official Train, Val, and Test splits, combining the Urban and Rural images within each split.

| Official split | Rural images | Urban images | Total |
| -------------- | -----------: | -----------: | ----: |
| Train          |         1366 |         1156 |  2522 |
| Val            |          992 |          677 |  1669 |
| Test           |          976 |          820 |  1796 |

The validation set is used for model selection and is not included in training. Final testing uses all 1796 images in the official Test split. Pixel-level ground-truth labels for this split are not publicly available, and predictions are evaluated through the official evaluation platform.

### ISPRS Potsdam

The dataset is partitioned using the original tile IDs. Of the 38 original tiles, tile `7_10` is excluded from the experiments. The remaining 37 tiles are assigned as follows.

```text
Training set (22 tiles)
2_11, 2_12, 3_10, 3_11, 3_12, 4_10, 4_11, 4_12,
5_10, 5_11, 5_12, 6_7, 6_8, 6_9, 6_10, 6_11,
6_12, 7_7, 7_8, 7_9, 7_11, 7_12

Validation set (1 tile)
2_10

Test set (14 tiles)
2_13, 2_14, 3_13, 3_14, 4_13, 4_14, 4_15,
5_13, 5_14, 5_15, 6_13, 6_14, 6_15, 7_13
```

### ISPRS Vaihingen

All 33 original images are used, with the splits defined by their area IDs.

```text
Training set (15 images)
1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 32, 34, 37

Validation set (1 image)
30

Test set (17 images)
2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38
```

## Data Preprocessing

- **Potsdam and Vaihingen:** The original images are assigned to their respective splits before cropping. Non-overlapping patches are extracted using a patch size and stride of 1024 × 1024. Training data include original, horizontally flipped, and vertically flipped views; validation and test data are not expanded through flipping. Padded regions are excluded from loss computation and evaluation. For Vaihingen, we use the COMPLETE ground-truth labels without boundary erosion.
- **LoveDA:** The original 1024 × 1024 RGB images are retained without offline cropping or flip augmentation.
- **Online training augmentation:** The input size is 512 × 512. Augmentations include Mosaic, random scaling, random cropping, and random rotation in multiples of 90°, followed by normalization using the ImageNet mean and standard deviation.
- **Validation and testing:** The input size is 1024 × 1024, and only normalization is applied. No test-time augmentation (TTA) is used.

The numbers of patches produced by offline preprocessing for Potsdam and Vaihingen are listed below. 

| Dataset         | Training patches | Validation patches | Test patches |
| --------------- | ---------------: | -----------------: | -----------: |
| ISPRS Potsdam   |             2376 |                 36 |          504 |
| ISPRS Vaihingen |              273 |                  6 |          113 |

## Training and Testing Settings

| Parameter                                       | Setting                                                      |
| ----------------------------------------------- | ------------------------------------------------------------ |
| Training epochs                                 | 105                                                          |
| Training batch size                             | 24                                                           |
| Validation/test batch size                      | 8                                                            |
| Optimizer                                       | AdamW with Lookahead                                         |
| Initial encoder learning rate                   | 6 × 10⁻⁵                                                     |
| Initial learning rate for the remaining modules | 6 × 10⁻⁴                                                     |
| Weight decay                                    | 0.01; no weight decay is applied to bias parameters          |
| Learning rate schedule                          | Cosine annealing with warm restarts; initial period: 15 epochs; period multiplier: 2 |
| Loss function                                   | Equally weighted sum of label-smoothed cross-entropy loss and Dice loss |
| Label smoothing factor                          | 0.05                                                         |
| Model selection                                 | Evaluate on the validation set after each epoch and test the model with the highest validation F1 |

## Evaluation Protocol

For Potsdam and Vaihingen, we report per-class F1, mean Intersection over Union (mIoU), mean F1 (mF1), and Overall Accuracy (OA). The mIoU and mF1 are macro-averages over the five classes excluding clutter, while OA is computed over valid pixels from all six classes.

LoveDA uses all seven classes, including background, for evaluation. In the original annotations, label 0 denotes invalid pixels, and labels 1–7 correspond to the seven valid classes. For data loading, label 0 is mapped to the ignore value 7, and labels 1–7 are mapped to class IDs 0–6. The background class participates in both training and evaluation, while invalid pixels are excluded. Test metrics are obtained from the official evaluation platform.
