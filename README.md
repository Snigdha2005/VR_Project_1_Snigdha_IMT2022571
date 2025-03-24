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
│   ├── dataset.csv
│   ├── face_crop
│   ├── face_crop_segmentation
│   └── img
└── 2
    └── img
```

## Methodology for Binary Classification Using Handcrafted Features and ML Classifiers

### **Dataset Loading and Preprocessing**
- The dataset consists of images of faces categorized as "with mask" and "without mask."
- Images are loaded using `cv2.imread()` from two directories (`./dataset/with_mask` and `./dataset/without_mask`).
- Each image is resized to a fixed resolution of **128x128** pixels using `cv2.resize()` to ensure uniformity in size across the dataset.
- Labels are assigned: **1 for "with mask"** and **0 for "without mask."**

### **Visualization**
- The first 5 images from both categories are displayed using `matplotlib.pyplot` to visually inspect and confirm the dataset's correctness.
- This provides an understanding of image variations and overall data quality.

### **Feature Extraction**
- Five different handcrafted feature extraction techniques are applied to represent image characteristics effectively:

1. **Histogram of Oriented Gradients (HOG)**:
    - Extracts edge and texture information using gradient orientations.
    - Parameters used: `pixels_per_cell=(4,4)` and `cells_per_block=(3,3)` for fine-grained details.

2. **Local Binary Pattern (LBP)**:
    - Captures local texture patterns using the relationship between neighboring pixels.
    - Parameters used: `radius=2` and `n_points=16`.

3. **Canny Edge Detection**:
    - Identifies object boundaries using the Canny edge detection algorithm.
    - Parameters: `threshold1=50` and `threshold2=150`.

4. **Scale-Invariant Feature Transform (SIFT)**:
    - Detects and describes key points using feature descriptors.
    - The top **50 keypoints** are retained, and descriptors are flattened for feature representation.

5. **Color Histogram**:
    - Extracts color distribution information from the HSV (Hue, Saturation, Value) color space.
    - 3D histogram with **32 bins per channel** is calculated using `cv2.calcHist()`.

- All the extracted features are concatenated using `np.hstack()` to form a single feature vector representing each image.

### **Data Preparation**
- The complete feature matrix (`X_features`) and labels (`y_labels`) are generated.
- The dataset is split into **80% training** and **20% testing** using `train_test_split()`.
- Features are scaled using `StandardScaler()` to normalize data for better classifier performance.

### **Model Training and Evaluation**
- Two machine learning classifiers are trained using the extracted features:

1. **Support Vector Machine (SVM)**:
    - A linear SVM (`SVC(kernel="linear")`) is used for binary classification.
    - It tries to find the optimal hyperplane for separation.

2. **Multilayer Perceptron (MLP)**:
    - A neural network with two hidden layers (**128 and 64 neurons**) is used.
    - ReLU activation and a maximum of **1000 iterations** are configured for training.

### **Evaluation Metrics**
- Both models are evaluated using:
  - **Accuracy Score**: Measures the ratio of correct predictions.
  - **Classification Report**: Provides metrics like precision, recall, and F1-score for both classes.
  - **Confusion Matrix**: Visualized using `seaborn.heatmap()` to observe misclassifications.

### **Comparison of Results**
- The final accuracy of both models is displayed and compared.
- The confusion matrix is analyzed to detect model behavior on both classes.

