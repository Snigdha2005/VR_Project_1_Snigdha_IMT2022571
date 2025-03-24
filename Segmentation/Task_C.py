import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.filters import sobel, threshold_sauvola
from skimage.color import label2rgb
from skimage.segmentation import watershed
from scipy import ndimage as nd
from skimage import measure, color, morphology
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

def dice_score(segmentation, ground_truth):
    intersection = np.sum(np.logical_and(segmentation, ground_truth))
    return (2.0 * intersection) / (np.sum(segmentation) + np.sum(ground_truth))

def apply_segmentation(image_path, method="otsu"):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "otsu":
        return gray, otsu_thresholding(gray)
    elif method == "adaptive":
        return gray, adaptive_thresholding(gray)
    elif method == "canny":
        return gray, canny_edge_detection(gray)
    elif method == "sobel":
        return gray, sobel_edge_detection(gray)
    elif method == "laplacian":
        return gray, laplacian_edge_detection(gray)
    elif method == "watershed":
        return gray, watershed_segmentation(gray)
    elif method == "morphology":
        return gray, morphological_segmentation(gray)
    elif method == "connected":
        return gray, connected_component_segmentation(gray)
    elif method == "kmeans":
        return gray, kmeans_segmentation(image)
    elif method == "local":
        return gray, local_thresholding(gray)
    else:
        raise ValueError("Invalid method. Choose from otsu, adaptive, canny, sobel, laplacian, watershed, morphology, or connected.")

# Method 1: Otsu's Thresholding
def otsu_thresholding(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# Method 2: Adaptive Thresholding
def adaptive_thresholding(gray):
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    return binary

# Method 3: Canny Edge Detection
def canny_edge_detection(gray):
    edges = cv2.Canny(gray, 50, 150)
    binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)[1]
    return binary

# Method 4: Sobel Edge Detection
def sobel_edge_detection(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=7)
    edges = cv2.magnitude(sobelx, sobely)
    edges = (edges / np.max(edges) * 255).astype(np.uint8)
    binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary

# Method 5: Laplacian Edge Detection
def laplacian_edge_detection(gray):
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edges = (np.abs(edges) / np.max(np.abs(edges)) * 255).astype(np.uint8)
    binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary

# Method 6: Watershed Segmentation
def watershed_segmentation(gray):
    blurred = cv2.GaussianBlur(gray, (101, 101), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    distance = nd.distance_transform_edt(binary)
    local_max = morphology.local_maxima(distance)
    markers, _ = nd.label(local_max)
    
    labels = watershed(-distance, markers, mask=binary)
    binary_labels = (labels > 0).astype(np.uint8) * 255
    return binary_labels

# Method 7: Morphological Segmentation
def morphological_segmentation(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    return cleaned

# Method 8: Connected Component Segmentation
def connected_component_segmentation(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labels = measure.label(binary, connectivity=1)
    labels = morphology.remove_small_objects(labels, min_size=200)
    binary_labels = (labels > 0).astype(np.uint8) * 255
    return binary_labels

def kmeans_segmentation(image, k=2):
    # Ensure the image is in BGR
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("K-Means requires a BGR color image.")
    
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    
    # K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image.shape)

    # Convert to grayscale
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding for binary output
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def local_thresholding(gray, window_size=1001):
    thresh_sauvola = threshold_sauvola(gray, window_size=window_size)
    binary = (gray > thresh_sauvola).astype(np.uint8) * 255
    return binary

def visualise_segmentation(face_path, gray, segmentation, ground_truth):
    image = cv2.imread(face_path)
    truth = cv2.imread(ground_truth)
    plt.figure(figsize=(18, 10))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Gray')

    plt.subplot(1, 4, 3)
    plt.imshow(segmentation, cmap='gray')
    plt.title('Segmentation')

    plt.subplot(1, 4, 4)
    plt.imshow(truth)
    plt.title('Ground_truth')

    plt.show()

def calculate_metrics(segmentation, ground_truth_path):
    ground_truth = cv2.imread(ground_truth_path, 0)
    ground_truth = cv2.resize(ground_truth, (segmentation.shape[1], segmentation.shape[0]), interpolation=cv2.INTER_NEAREST)

    ground_truth = (ground_truth > 0).astype(np.uint8)
    segmentation = (segmentation > 0).astype(np.uint8)

    iou = np.sum(np.logical_and(segmentation, ground_truth)) / np.sum(np.logical_or(segmentation, ground_truth))
    f1 = f1_score(ground_truth.flatten(), segmentation.flatten())
    accuracy = accuracy_score(ground_truth.flatten(), segmentation.flatten())
    dice_score = 2 * np.sum(np.logical_and(segmentation, ground_truth)) / (np.sum(segmentation) + np.sum(ground_truth))
    jaccard = jaccard_score(ground_truth.flatten(), segmentation.flatten())

    return iou, f1, accuracy, dice_score, jaccard

def main():
    input_path = './MSFD/1/face_crop'
    ground_truth_path = './MSFD/1/face_crop_segmentation'
    methods = ["otsu", "adaptive", "canny", "sobel", "laplacian", "watershed", "morphology", "connected", "kmeans", "local"]

    metrics = {method: {"iou": [], "f1": [], "accuracy": [], "dice": [], "jaccard": []} for method in methods}
    i = 0
    for face_filename in os.listdir(input_path):
        face_path = os.path.join(input_path, face_filename)
        ground_truth = os.path.join(ground_truth_path, face_filename)
        if (i % 250) == 0:
            if os.path.isfile(face_path) and os.path.isfile(ground_truth):
                for method in methods:
                    gray, segmentation = apply_segmentation(face_path, method)
                    iou, f1, accuracy, dice_score, jaccard = calculate_metrics(segmentation, ground_truth)
                    print(f"{face_filename} {method}: IoU={iou:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}, Dice={dice_score:.4f}, Jaccard={jaccard:.4f}")

                    metrics[method]["iou"].append(iou)
                    metrics[method]["f1"].append(f1)
                    metrics[method]["accuracy"].append(accuracy)
                    metrics[method]["dice"].append(dice_score)
                    metrics[method]["jaccard"].append(jaccard)
                    visualise_segmentation(face_path, gray, segmentation, ground_truth)
        i = i + 1

    print("\nAverage Metrics for each Method:")
    for method in methods:
        if metrics[method]["iou"]:
            avg_iou = np.mean(metrics[method]["iou"])
            avg_f1 = np.mean(metrics[method]["f1"])
            avg_accuracy = np.mean(metrics[method]["accuracy"])
            avg_dice = np.mean(metrics[method]["dice"])
            avg_jaccard = np.mean(metrics[method]["jaccard"])
            print(f"{method}: IoU={avg_iou:.4f}, F1={avg_f1:.4f}, Accuracy={avg_accuracy:.4f}, Dice={avg_dice:.4f}, Jaccard={avg_jaccard:.4f}")
        else:
            print(f"{method}: No data available.")
    
    print("\nMaximum IoU for each Method:")
    for method in methods:
        if metrics[method]["iou"]:
            max_iou = np.max(metrics[method]["iou"])
            print(f"{method}: Max IoU={max_iou:.4f}")
        else:
            print(f"{method}: No data available.")


if __name__ == "__main__":
    main()

