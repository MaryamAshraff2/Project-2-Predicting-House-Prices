# House Price Prediction using TensorFlow Decision Forests

This project is a machine learning model designed to predict house prices based on various features of houses. It utilizes **TensorFlow Decision Forests** (TF-DF) to implement a **Random Forest** regression model. The dataset used for this project is based on house sales data, and the target variable is `SalePrice`.

## Table of Contents
- [Installation](#installation)
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/MaryamAshraff2/Project-2-Predicting-House-Prices.git
    cd Project-2-Predicting-House-Prices
    ```

2. Install the required dependencies:
    ```bash
    pandas
    seaborn
    matplotlib
    tensorflow
    tensorflow-decision-forests
    ```

3. Launch the notebook or run the script.

## Project Overview

The aim of this project is to predict house prices using a Random Forest model implemented with TensorFlow Decision Forests. This project involves:

- Data loading and exploration.
- Preprocessing and cleaning.
- Feature selection and analysis.
- Training a regression model using Random Forest.
- Evaluating the model on unseen data.

## Data Preprocessing

1. **Loading Data**: The dataset is loaded into a Pandas DataFrame.
2. **Cleaning Data**: The `Id` column is dropped as it's not necessary for prediction. The dataset is split into training and validation sets.
3. **Feature Analysis**: Numerical features are explored to understand the distribution of data, and the target variable (`SalePrice`) is visualized.
4. **Conversion for TensorFlow**: The dataset is converted to TensorFlow's `tf.data.Dataset` format to optimize the training process.

## Model Training

The model chosen is a **Random Forest Regressor** from TensorFlow Decision Forests, which is suitable for predicting continuous values like house prices.

- **Task**: Regression
- **Metric**: Mean Squared Error (MSE)

The training process includes:
- Fitting the Random Forest model on the training dataset.
- Visualizing the decision trees and training progress.

## Evaluation

The model is evaluated on both the **Out-of-Bag (OOB)** data and the validation dataset. The evaluation metrics include:
- **Root Mean Squared Error (RMSE)**
- **Mean Decrease in AUC** for feature importance.

## Usage

To run the project, follow these steps:

1. Upload your house price dataset (`train_house_price.csv`) to the working directory.
2. Open the Python script or Jupyter Notebook.
3. Run the code to train the model and evaluate its performance.

You can visualize:
- The distribution of house prices.
- Numerical features in the dataset.
- The model's training progress and evaluation metrics.
- Variable importance to understand which features affect house prices the most.

## Repository Structure

```bash
house-price-prediction/
├── Project_2_Predicting_House_Prices.ipynb  # Main Google Colab file
├── train_house_price.csv                    # Dataset (you can upload your own)
├── README.md                                # Project README file
├── requirements.txt                         # List of dependencies
└── .gitignore                               # Git ignore file
```
## Acknowledgments

- Thanks to Kaggle for providing the House Price dataset.
- Inspiration and tutorials from various data science and machine learning resources.

## Contact

If you have any questions or suggestions, please contact me at maryamaff2001@gmail.com
