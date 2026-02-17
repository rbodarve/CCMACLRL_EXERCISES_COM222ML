# CCMACLRL Exercises - Introduction to Machine Learning

A comprehensive collection of machine learning exercises for the Introduction to Machine Learning (COM222ML) course by CCMACLRL (Crimson College of Management and Advanced Learning Resources).

This repository contains practical implementations of core machine learning concepts including data preprocessing, model training, classification, regression, and natural language processing.

## Table of Contents

- [Exercises Overview](#exercises-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [License](#license)
- [Credit](#credit)

## Exercises Overview

### Exercise 1: Data Loading and Exploration
Introduction to data handling using Jupyter notebooks. Learn to load, explore, and understand datasets.

### Exercise 2: K-Nearest Neighbors (KNN) Classification
Machine learning pipeline with KNN classifier applied to the Iris dataset. Includes model training, evaluation, and prediction.

### Exercise 3: KNN Classification with Personality Dataset
Build a KNN model to classify personality types (introvert/extrovert). Performance optimization through parameter tuning.

### Exercise 4: Simple Linear Regression
Study the linear relationship between years of experience and salary. Includes model equation derivation and error calculation.

### Exercise 5: Multiple Linear Regression with House Prices
Predict house prices using multiple features. Includes feature selection, scaling, and model evaluation with R-squared metric.

### Exercise 5B: Advanced Multiple Linear Regression
House price prediction with data preprocessing, feature engineering with dummy variables, multicollinearity analysis, and statistical model summary.

### Exercise 6: Data Preprocessing and Feature Engineering
Focus on data cleaning, transformation, and feature engineering techniques.

### Exercise 7: Hate Speech Classification using Multinomial Naive Bayes
Text classification pipeline including NLP preprocessing (tokenization, lemmatization, stop word removal), TF-IDF vectorization, and model evaluation.

### Exercise 8: Computer Vision with Image Processing
Image processing and analysis using OpenCV and related techniques.

### Exercise 9: Advanced Machine Learning Techniques
Implementation of decision trees, random forests, RFE (Recursive Feature Elimination), and KNN ensemble methods.

## Project Structure

```
CCMACLRL_EXERCISES_COM222ML/
├── Exercise 1/
│   └── main.ipynb
│   └── requirements.txt
├── Exercise 2/
│   ├── Exercise2.ipynb
│   ├── iris.csv
│   └── requirements.txt
├── Exercise 3/
│   ├── Exercise3.ipynb
│   └── requirements.txt
├── Exercise 4/
│   ├── Exercise4.ipynb
│   ├── salary.csv
│   └── requirements.txt
├── Exercise 5/
│   ├── Exercise5.ipynb
│   └── requirements.txt
├── Exercise 5b/
│   ├── Exercise5B.ipynb
│   ├── house_prices.csv
│   └── requirements.txt
├── Exercise 6/
│   ├── Exercise6_.ipynb
│   └── submission_file.csv
├── Exercise 7/
│   ├── Exercise7.ipynb
├── Exercise 8/
│   ├── Exercise8.ipynb
├── Exercise 9/
│   ├── exercise9 - Navida.ipynb
│   ├── Exercise9-Odarve.ipynb
│   ├── Montaniel_Exercise9.ipynb
│   ├── submission_knn.csv
│   ├── submission_lr.csv
│   └── resources/
│       ├── DecisionTree.csv
│       ├── RFE.csv
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
└── README.md
```

## Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- pip or conda package manager

## Important Note

Some URL links and external resources referenced in the notebooks may not be accessible or may have changed. Please verify links before use and adjust them as needed for your environment.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/rbodarve/CCMACLRL_EXERCISES_COM222ML.git
cd CCMACLRL_EXERCISES_COM222ML
```

### 2. Install Dependencies

Each exercise folder contains a `requirements.txt` file. Install the required packages:

```bash
pip install -r Exercise\ 1/requirements.txt
```

Or install all dependencies at once:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter nltk seaborn statsmodels
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then navigate to your desired exercise folder and open the `.ipynb` file.

## Requirements

Core Python packages used across exercises:

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning library
- **nltk**: Natural language processing toolkit
- **statsmodels**: Statistical modeling
- **opencv-python**: Computer vision library
- **openpyxl**: Excel file handling

## License

This project is provided for educational purposes.

## Credit

README template structure inspired by [Awesome README](https://github.com/matiassingers/awesome-readme) by Matias Singers.
