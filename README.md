## Overview
This project focuses on developing a Convolutional Neural Network (CNN) model to detect lung cancer from medical images, specifically CT scans or X-rays. The model is designed to classify lung images into three categories: benign, malignant, and normal. By leveraging advanced preprocessing techniques, such as image resizing, grayscale conversion, and normalization, the project ensures high-quality input data for the model. Additionally, random oversampling is applied to address class imbalance, ensuring that the model is trained on a balanced dataset.

The CNN architecture is built using TensorFlow and Keras, incorporating multiple Conv2D and MaxPooling2D layers for feature extraction, followed by fully connected Dense layers with Dropout for regularization. The model is trained and evaluated on The IQ-OTHNCCD Lung Cancer Dataset, achieving an impressive accuracy of 99%. 

## Key Features
Dataset: Uses the IQ-OTHNCCD Lung Cancer Dataset, containing images of benign, malignant, and normal lung cases.
Preprocessing: Resizes images to 128x128, converts them to grayscale, normalizes pixel values, and applies random oversampling to handle class imbalance.
Model: A CNN architecture with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.
Evaluation: Evaluates model performance using classification reports, confusion matrices, and training/validation accuracy/loss plots.
Visualization: Displays correctly classified images with true and predicted labels.

## Requirements
To run this project, you need the following libraries:
Python 3.x
TensorFlow
Keras
OpenCV
NumPy
Scikit-learn
Matplotlib
Plotly
Pandas
Imbalanced-learn (for oversampling)

## Dataset
The dataset used in this project is The IQ-OTHNCCD Lung Cancer Dataset, which contains images of:
Benign cases
Malignant cases
Normal cases

The dataset is available on Kaggle or other public repositories.

## Project Workflow
# Data Collection: 
Load images from the dataset.

# Preprocessing:
Resize images to 128x128.
Convert images to grayscale.
Normalize pixel values to [0, 1].
Apply random oversampling to balance the dataset.

# Model Building:
Define a CNN model with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.
Compile the model using the Adam optimizer and categorical crossentropy loss.

# Training: 
Train the model on the preprocessed dataset.

Evaluation:
Generate a classification report.
Plot a confusion matrix.
Visualize training and validation accuracy/loss.

# Visualization:
Display correctly classified images with true and predicted labels.

## Results
The CNN model achieved an impressive accuracy on the test set.
The classification report provides precision, recall, and F1-score for each class.
The confusion matrix visualizes the model's performance in classifying benign, malignant, and normal cases.
Training and validation plots show the model's learning progress over epochs.
