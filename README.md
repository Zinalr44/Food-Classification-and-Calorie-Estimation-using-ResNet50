# Food-Classification-and-Calorie-Estimation-using-ResNet50
Overview
This project implements a food classification system using the ResNet50 architecture, pre-trained on ImageNet and fine-tuned on the Food-101 dataset. The system classifies images into specific food categories and estimates the calorie content of the classified food.

Dataset
The project uses the Food-101 dataset, which consists of 101 food categories. The dataset is split into training and testing sets.

Data Preparation
Data Check: Ensures the existence of the dataset and meta files.
Data Visualization: Displays sample images from each food category.
Dataset Split: Splits the dataset into training and testing folders based on provided metadata.
Mini Dataset: Creates smaller subsets of the dataset for quick experimentation with fewer classes.
Model Architecture
Pre-trained Model: Uses ResNet50 pre-trained on ImageNet.
Fine-Tuning: Adapts ResNet50 for the specific task of food classification with a custom top layer.
Regularization: Applies dropout and L2 regularization to prevent overfitting.
Training
Data Augmentation: Applies image transformations to improve model generalization.
Model Training: Trains the model on a mini dataset for quick validation and then on a larger dataset.
Callbacks: Utilizes ModelCheckpoint to save the best model and CSVLogger to log training history.
Evaluation
Accuracy and Loss: Plots learning curves for training and validation accuracy and loss.
Model Testing: Loads the best model and performs predictions on test images.
Calorie Estimation
Calorie Dictionary: Provides calorie information for each food category.
Prediction with Calories: Predicts the food category and estimates the calorie content of test images.
Usage
Dataset Preparation:
Ensure the Food-101 dataset and meta files are correctly placed.
Use prepare_data() to organize the dataset into training and testing directories.
Training:
Run the script to train the model on the dataset.
Fine-tune the model on a subset of classes if needed.
Prediction:
Use the predict_class_and_calories() function to classify images and estimate calories.
Future Work
Extend the model to handle more food categories.
Experiment with other pre-trained models for comparison.
Implement a web or mobile application for real-time food classification and calorie estimation.
Results
Model Performance: Achieved high accuracy on food classification tasks.
Calorie Estimation: Successfully estimates calories based on classified food categories.
Acknowledgments
The creators of the Food-101 dataset.
TensorFlow and Keras communities for their powerful tools and resources.
