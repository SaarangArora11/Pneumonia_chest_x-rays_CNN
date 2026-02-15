#Pneumonia Detection from Chest X-Rays

A Convolutional Neural Network (CNN) designed to classify chest X-ray images as either "Normal" or "Pneumonia-infected".

This project demonstrates the application of Computer Vision in medical diagnostics, aiming to assist radiologists by automating the detection of lung opacities associated with pneumonia.
## Objective

The goal was to build a robust binary classifier for medical imaging.
Problem: Pneumonia is a life-threatening condition that requires rapid diagnosis. Manual review of X-rays is time-consuming and subject to human error.
Solution: An automated deep learning model that can flag potential cases with high sensitivity.
Focus: Prioritizing Recall (Sensitivity) over Precision to ensure valid cases are not missed (minimizing False Negatives).

## Key Concepts & Skills

  1. Convolutional Neural Networks (CNNs): Building deep networks to automatically learn spatial hierarchies of features (edges -> textures -> lung shapes).
  2. Data Augmentation: artificially increasing the diversity of the training set (zooming, shearing, flipping) to prevent overfitting on a limited medical dataset.
  3. Medical Image Analysis: Handling grayscale image data and understanding class imbalance (more pneumonia cases than normal).
  4. Evaluation Metrics: Analyzing the Confusion Matrix, Precision, Recall, and F1-Score.

## Methodology / Architecture
1. Data Preprocessing

    * Rescaling: Pixel values were normalized to the [0, 1] range (1/255) for faster convergence.
    * Augmentation: Applied `ImageDataGenerator` to introduce variations (shear, zoom, horizontal flip) to make the model robust to different X-ray positionings.
    * Resizing: Images were resized to a fixed input shape (e.g., 150×150 or 224×224) to match the network input.

2. Model Architecture (CNN)

I designed a custom CNN architecture (or used Transfer Learning, e.g., VGG16/ResNet):

* Convolutional Layers: To extract feature maps (detecting edges of ribs, lungs, and opacities).
* Max Pooling: To downsample the features and reduce computational cost.
* Batch Normalization: To stabilize and accelerate training.
* Dropout: To prevent overfitting by randomly disabling neurons during training.
* Dense Head: Fully connected layers with a final `Sigmoid` activation for binary classification (0 = Normal, 1 = Pneumonia).

3. Training

* Optimizer: Adam (adaptive learning rate).
* Loss Function: `BinaryCrossentropy`.
* Callbacks: Used `EarlyStopping` and `educeLROnPlateau` to prevent overfitting and fine-tune the learning rate dynamically.

## Code Highlight

Here is the snippet defining the core CNN blocks:
```Python

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

## Results

The model achieves high recall on the test set.

  * Recall (Pneumonia): 94.35% High sensitivity, ensuring most sick patients are correctly identified.

## Dependencies

```Python 3.x
    TensorFlow / Keras
    NumPy, Pandas
    Matplotlib / Seaborn (for visualization)
    OpenCV (cv2)
```

## How to Run

  Clone the repository.
  Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle.
  Update the data directory paths in the notebook.
  Run chest-x-ray-cv.ipynb.

## Future Improvements
* Explainable AI (Grad-CAM): Implement Gradient-weighted Class Activation Mapping to visualize where the model is looking in the X-ray (heatmap overlay).
* Transfer Learning: Compare performance against heavier models like DenseNet121 (often used in medical imaging).
* Multi-class Classification: Extend the model to detect Viral vs. Bacterial Pneumonia.

## References / Credits

Dataset: Chest X-Ray Images (Pneumonia) - Kermany et al.
Research: Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning (Cell, 2018)
