# Data Preprocessing for Machine Learning - Iris Dataset

## Objective
To preprocess raw data and make it suitable for machine learning models.

## Dataset
The Iris dataset contains measurements of flower features such as sepal length, sepal width, petal length, and petal width, along with species classification.

## Steps Performed
- Loaded dataset using pandas
- Checked for missing values
- Encoded categorical variable (species) using Label Encoding
- Standardized numerical features using StandardScaler
- Split dataset into training and testing sets

## Output
A processed dataset (`processed_iris.csv`) with:
- Scaled numerical features
- Encoded target variable

## Tools Used
- Python
- Pandas
- Scikit-learn