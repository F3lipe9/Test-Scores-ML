# Student Admissions ML Predictor

A machine learning project that predicts student admission outcomes based on admission test scores and high school percentages using Logistic Regression and Random Forest models.

## Overview

This project analyzes student admission data to predict whether a student will be accepted or rejected based on their academic performance. The model uses features like admission test scores, high school percentages, and engineered features to make predictions.

## Features

- **Data Cleaning**: Removes invalid entries, null values, and outliers
- **Feature Engineering**: Creates additional features including:
  - Score × Percentage interaction
  - Score - Percentage difference
  - Score/Percentage ratio
  - Squared values for both metrics
- **Multiple Models**: Implements both Logistic Regression and Random Forest Classifier
- **Data Preprocessing**: Uses StandardScaler for feature normalization

## Dataset

**Input**: `student_admission_record_dirty.csv`
- Contains student information including name, city, gender, age, test scores, high school percentages, and admission status

**Output**: `output.csv`
- Cleaned and processed dataset with engineered features

## Data Cleaning Process

The cleaning pipeline:
1. Removes demographic columns (Name, City, Gender, Age)
2. Filters out null values in key columns
3. Removes invalid score ranges (< 0 or > 100)
4. Eliminates outlier combinations:
   - Students with < 50% scores who were accepted
   - Students with > 90% scores who were rejected
5. Removes duplicate entries

## Models

### Logistic Regression
- Uses a pipeline with StandardScaler
- Max iterations: 9999
- Predicts binary admission outcome

### Random Forest Classifier
- 300 estimators
- Random state: 42
- Ensemble method for improved accuracy

## Requirements

```
pandas
scikit-learn
```

## Installation

```bash
pip install pandas scikit-learn
```

## Usage

Run the main script:

```bash
python main.py
```

The script will:
1. Load and clean the data
2. Engineer features
3. Save cleaned data to `output.csv`
4. Train both models
5. Print accuracy scores for each model

## Model Evaluation

The project uses an 80/20 train-test split with stratification to ensure balanced class distribution. Accuracy scores are printed for both models.

## Project Structure

```
admissionsML/
├── main.py                              # Main ML pipeline
├── student_admission_record_dirty.csv   # Raw input data
├── output.csv                           # Cleaned data
└── README.md                            # Project documentation
```

## Future Improvements

- Cross-validation for more robust evaluation
- Hyperparameter tuning
- Additional evaluation metrics (precision, recall, F1-score)
- Feature importance analysis
- Model persistence (save/load trained models)
