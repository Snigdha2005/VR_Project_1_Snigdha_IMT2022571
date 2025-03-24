# Face Mask Classification and Segmentation

## Introduction

This project aims to develop a comprehensive computer vision solution for classifying and segmenting face masks in images using both traditional machine learning (ML) classifiers and deep learning techniques. The primary objectives include:

- **Binary Classification with ML Classifiers**: Classify faces as "with mask" or "without mask" using handcrafted features and machine learning classifiers like Support Vector Machines (SVM) and Neural Networks.
- **Binary Classification with CNN**: Design and train a Convolutional Neural Network (CNN) for the same classification task, applying different hyperparameter variations to optimize performance.
- **Region-Based Segmentation**: Implement traditional region-based segmentation techniques (e.g., thresholding, edge detection) to segment face masks from images identified as "with mask."
- **Mask Segmentation with U-Net**: Train a U-Net model for accurate mask region segmentation and compare its performance with traditional segmentation methods using metrics like Intersection over Union (IoU) and Dice score.

## Dataset

### **Binary Classification Dataset**
- **Source**: [Face Mask Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)
- **Structure**: The dataset contains two folders:
  - `with_mask`: Contains images of individuals wearing face masks.
  - `without_mask`: Contains images of individuals without face masks.
- **Total Images**: 4,095 images
  - `with_mask`: 2,165 images
  - `without_mask`: 1,930 images

### **Segmentation Dataset**
- **Source**: [MFSD Dataset](https://github.com/sadjadrz/MFSD)
- **Structure**:
  - `1` and `2`: Each contains an `img` folder with images of people with and without face masks.
  - `face_crop`: Cropped face images from the `img` folders.
  - `face_crop_segmentation`: Ground truth images representing mask segmentation.
  - `dataset.csv`: Metadata for the dataset.

```
MSFD
├── 1
│   ├── dataset.csv
│   ├── face_crop
│   ├── face_crop_segmentation
│   └── img
└── 2
    └── img

```

