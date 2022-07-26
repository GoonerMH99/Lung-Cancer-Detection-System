# Lung-Cancer-Detection-System
Web Application developed to detect cancer nodules in CT-Scans image dataset using deep learning model "CNN", this project was submitted as a graduation project at the "Faculty of Computers and Artificial Intelligence-Cairo University" 2021-2022

## Table of Contents

* Technologies

* Dataset

* Image Format

* Pre-Processing

* CNN Model


## Technologies

* Python 3
* Keras, TensorFlow, SimpleITK, OpenCV, Numpy, Pydicom and Matplotlib
* HTML, CSS, Java Script, Xampp and Flask.
* Google Colab.

## Dataset
Dataset from Kaggle's competition "Data Science Bowl 2017"

Dataset Files:
- stage1.7z - contains all images for the first stage of the competition, including both the training and test set.
- stage1_labels.csv - contains the cancer ground truth for the stage 1 training set images.

## Image Format
In the "stage1.7z" directory, there are 1,595 directories one for each patient each directory is named after a patient id, inside each patient directory there are a number of CT-Scan slices varying from 130 to 280 slices, each slice is an image saved in the "Dicom" format as the figure below.

![image](https://user-images.githubusercontent.com/101012808/181118683-99bf931d-873e-4f81-88a2-f9eabe391b5d.png)

## Pre-Processing
Slices were segmented, resized and reshaped to be in a suitable format for machine learning models.

![image](https://user-images.githubusercontent.com/101012808/181120522-f0ce0688-8253-44a3-9e0a-c0e54d4db387.png)

## CNN Model

The CNN model used consisted of 2 convolutional layers and hidden layer of 100 nodes and output layer of 2 nodes resembling the 2 classes (Cancerous & Non-Cancerous), this model was used to solve the image classification problem.

## Web Application Frontend and Backend

![image](https://user-images.githubusercontent.com/101012808/181122629-eeeb087a-647d-4ba6-b77e-26b0951b599e.png)
