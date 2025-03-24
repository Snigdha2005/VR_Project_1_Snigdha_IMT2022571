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

## **Task A**

### **Methodology**

#### **Dataset Loading and Preprocessing**
- The dataset consists of images of faces categorized as "with mask" and "without mask."
- Images are loaded using `cv2.imread()` from two directories (`./dataset/with_mask` and `./dataset/without_mask`).
- Each image is resized to a fixed resolution of **128x128** pixels using `cv2.resize()` to ensure uniformity in size across the dataset.
- Labels are assigned: **1 for "with mask"** and **0 for "without mask."**

#### **Visualization**
- The first 5 images from both categories are displayed using `matplotlib.pyplot` to visually inspect and confirm the dataset's correctness.
- This provides an understanding of image variations and overall data quality.

#### **Feature Extraction**
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

#### **Data Preparation**
- The complete feature matrix (`X_features`) and labels (`y_labels`) are generated.
- The dataset is split into **80% training** and **20% testing** using `train_test_split()`.
- Features are scaled using `StandardScaler()` to normalize data for better classifier performance.

#### **Model Training and Evaluation**
- Two machine learning classifiers are trained using the extracted features:

1. **Support Vector Machine (SVM)**:
    - A linear SVM (`SVC(kernel="linear")`) is used for binary classification.
    - It tries to find the optimal hyperplane for separation.

2. **Multilayer Perceptron (MLP)**:
    - A neural network with two hidden layers (**128 and 64 neurons**) is used.
    - ReLU activation and a maximum of **1000 iterations** are configured for training.

#### **Evaluation Metrics**
- Both models are evaluated using:
  - **Accuracy Score**: Measures the ratio of correct predictions.
  - **Classification Report**: Provides metrics like precision, recall, and F1-score for both classes.
  - **Confusion Matrix**: Visualized using `seaborn.heatmap()` to observe misclassifications.

### **Results and Observations**

- **SVM Performance**: The SVM classifier achieved an accuracy of **93.04%**. Precision and recall scores for both classes were balanced, indicating reliable classification performance. The classifier performed well for both mask and no-mask cases.
  
- **MLP Performance**: The MLP classifier outperformed SVM slightly with an accuracy of **93.77%**. Notably, MLP showed superior performance in classifying faces **without masks**, achieving a higher F1-score and recall compared to SVM.

- **Class-Specific Observations**:
  - For the "with mask" class, both SVM and MLP showed similar prediction accuracy with a high recall and precision.
  - For the "without mask" class, MLP demonstrated better detection, reducing false negatives and achieving a recall of **0.94** compared to SVM's **0.92**.

- **Conclusion**: MLP's additional computational capacity enabled it to extract deeper patterns within the feature space, leading to superior classification performance for difficult cases.

These findings suggest that MLP may be a better choice when the primary goal is to minimize false negatives in mask detection scenarios.

## **Task B**

### **Methodology**

#### **1. Data Collection and Preprocessing**
- **Dataset**: The dataset consists of images categorized into two classes: `with_mask` and `without_mask`.
- **Image Preprocessing**:
  - All images were resized to **128x128** pixels for uniformity.
  - Images were normalized by dividing pixel values by **255.0** to scale them between **0 and 1**.
- **Labeling**:
  - `1` for **with mask**
  - `0` for **without mask**

#### **2. Data Splitting**
- The dataset was split into:
  - **70%** for Training
  - **15%** for Validation
  - **15%** for Testing
- `train_test_split()` was used with `random_state=42` for reproducibility.

#### **3. Model Design and Hyperparameters**
- A Convolutional Neural Network (CNN) was designed with the following layers:
  - **Conv2D (32 filters)** with **ReLU** activation
  - **MaxPooling2D (2x2)**
  - **Conv2D (64 filters)** with **ReLU** activation
  - **MaxPooling2D (2x2)**
  - **Conv2D (128 filters)** with **ReLU** activation
  - **MaxPooling2D (2x2)**
  - **Flatten** layer
  - **Dense (64 neurons)** with **ReLU**
  - **Dropout (50%)** for regularization
  - **Dense (1 neuron)** with **Sigmoid/Tanh** activation for binary classification

- **Optimizers**: Adam and SGD were used with varying learning rates and batch sizes.
- **Loss Function**: Binary Crossentropy.
- **Evaluation Metrics**: Accuracy and Validation Accuracy.

#### **4. Hyperparameter Variations**
Four sets of hyperparameters were experimented with:

| Model | Activation | Optimizer | Learning Rate | Batch Size |
|---------|-------------|-----------|----------------|-----------|
| 1      | Sigmoid    | Adam      | 0.001         | 32         |
| 2      | Sigmoid    | Adam      | 0.0005        | 64         |
| 3      | Sigmoid    | SGD       | 0.001         | 32         |
| 4      | Tanh       | Adam      | 0.001         | 32         |

#### **5. Training and Evaluation**
- Each model was trained for **10 epochs** using the selected hyperparameters.
- The training used `tf.data.Dataset` for efficient data loading and batch processing.
- Accuracy and Loss metrics were plotted for both training and validation.
- The model with the highest validation accuracy was selected for further testing.
- Final evaluation was conducted using the test dataset, and predictions were compared with ground truth using a **classification report** and a **confusion matrix**.

### **Observations and Results**

1. **Training Performance:**
   - Model 1 with sigmoid activation, Adam optimizer, and a learning rate of 0.001 achieved the highest accuracy with rapid convergence.
   - The validation accuracy consistently improved across epochs, reaching 95.28% by the end of training.
   - Model 2 with a lower learning rate (0.0005) also showed good performance, though slightly lower than Model 1.

2. **Effect of Optimizer Choice:**
   - Models trained using the Adam optimizer generally outperformed the SGD-based model (Model 3).
   - The slower convergence and lower accuracy observed with the SGD optimizer suggest that Adam's adaptive learning rates were beneficial for this task.

3. **Activation Function Impact:**
   - The models using the sigmoid activation function showed more stable learning and higher accuracy compared to the model using tanh (Model 4).
   - Model 4 exhibited performance instability, particularly with a significant spike in validation loss at epoch 4, indicating potential issues with saturation and gradient vanishing.

4. **Validation and Test Performance:**
   - The best model (Model 1) achieved a validation accuracy of 95.28% and a test accuracy of 96%.
   - Precision, recall, and F1-scores were balanced across both classes (with_mask and without_mask), suggesting no significant class imbalance or overfitting.

- **Best Model Configuration:**
  - Activation Function: Sigmoid
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Batch Size: 32

- **Test Set Performance:**
  - Accuracy: **96%**
  - Precision: **0.95 (without_mask), 0.96 (with_mask)**
  - Recall: **0.96 (without_mask), 0.96 (with_mask)**
  - F1-Score: **0.95 (without_mask), 0.96 (with_mask)**

- **Conclusion:**
  - The best-performing model demonstrated high classification accuracy with excellent precision and recall for both classes.
  - The consistent performance across training, validation, and test sets confirms the model's generalizability and robustness.
  - Further improvements could involve experimenting with additional regularization techniques or fine-tuning hyperparameters further.

## **Comparison of Task A and Task B Results and Observations**

#### **1. Accuracy**  
- **Task A (SVM)**: 93.04%  
- **Task A (MLP)**: 93.77%  
- **Task B (CNN)**: 96%  
- **Observation**: CNN outperformed both SVM and MLP, leveraging its ability to extract complex spatial features.

#### **2. Performance on Mask Detection**  
- **Task A (MLP)**: Better recall for "without mask" class.  
- **Task B (CNN)**: Balanced precision and recall for both classes.  
- **Observation**: CNN provided more reliable predictions across both categories.

#### **3. Feature Extraction**  
- **Task A**: Handcrafted features (HOG, LBP, SIFT).  
- **Task B**: Automatic feature extraction using CNN layers.  
- **Observation**: CNN’s automated feature extraction led to improved classification accuracy.

#### **4. Training Time**  
- **Task A**: Faster with SVM and MLP.  
- **Task B**: Longer training time due to CNN’s complexity.  
- **Observation**: CNN demands more resources but results in better performance.

#### **5. Generalizability**  
- **Task A**: Moderate generalizability with handcrafted features.  
- **Task B**: High generalizability, with stable performance across training, validation, and test sets.  
- **Observation**: CNN is preferable for large-scale applications.

#### **Conclusion**  
- **For Simplicity and Speed**: SVM or MLP is recommended.  
- **For High Accuracy and Robustness**: CNN is the better choice.