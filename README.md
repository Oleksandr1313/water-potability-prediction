
# Water Potability Prediction

Water Potability Prediction using TensorFlow and pandas. A machine learning model assesses water quality, providing insights into its potability status.


## Overview

This project utilizes machine learning techniques to predict the potability of water based on various water quality attributes. It employs a neural network model implemented using TensorFlow and performs data preprocessing using pandas.
## Features

- Data Preprocessing: The provided dataset is loaded using pandas. Missing values are filled with the mean of each column, ensuring a clean and complete dataset.

- Neural Network Model: A neural network model is implemented using TensorFlow, comprising a dense hidden layer with ReLU activation and an output layer with a sigmoid activation for binary classification.

- Training and Evaluation: The model is trained on a portion of the dataset and evaluated on the remaining data. The binary crossentropy loss function is used for optimization.
## Data source

The water quality dataset used in this project is obtained from the official Kaggle website. The dataset, titled "Water Quality - Potability," can be found at [Kaggle - Water Quality - Potability](https://www.kaggle.com/adityakadiwal/water-potability).

Please ensure compliance with Kaggle's terms of use and licensing for the dataset.
