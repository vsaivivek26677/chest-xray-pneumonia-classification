# Dataset Instructions

This folder is reserved for the chest X-ray image dataset used by this project.

The dataset files are **not committed to GitHub**. To run the notebook locally, download the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:

```text
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

A Kaggle account may be required to download the dataset.

## Required Dataset Folder

After downloading and extracting the dataset, place the `chest_xray/` folder inside this `data/` directory.

## Expected Local Structure

```text
data/
├── README.md
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

Only `README.md` should be committed to GitHub. The `chest_xray/` dataset folder should remain local.

## Expected Dataset Format

The project expects a binary image-classification dataset with two class folders:

```text
NORMAL
PNEUMONIA
```

The notebook expects JPEG/JPG image files inside the class folders.

## Expected Path

Point the notebook at this folder by setting the dataset path in its second code cell:

```python
base_dir = '../data/chest_xray'
```

## How the Dataset Is Used

The notebook uses the dataset for:

- Exploratory image analysis
- Image loading and visual inspection
- Class balance checks
- HOG feature extraction for classical machine learning
- Random Forest and SVM classification
- Custom CNN training with PyTorch
- Transfer learning with ResNet18

## Notes

- The dataset is intentionally excluded from this repository to keep the GitHub project lightweight and to avoid redistributing third-party data.
- If the extracted folder has a different name, rename it to `chest_xray` or update the notebook path.
- Do not commit the image dataset, generated model checkpoints, or local training artifacts to GitHub.
