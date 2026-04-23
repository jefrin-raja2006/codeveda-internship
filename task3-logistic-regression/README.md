# Task 3: Logistic Regression — Sentiment Classification

## 📌 Overview

This project implements a Logistic Regression model to classify text data into sentiment categories using a real-world dataset.
The goal is to predict whether a given text expresses a **positive** or **negative** sentiment.

---

## 📊 Dataset

* Type: Text (Sentiment Dataset)
* Features:

  * **Text** → input data
  * **Sentiment** → target variable (Positive / Negative / Neutral)

For this task:

* Only **Positive** and **Negative** sentiments are used
* **Neutral** values are removed to maintain binary classification

---

## ⚙️ Workflow

### 1. Data Loading

* Dataset is loaded using pandas
* Separator is auto-detected for flexibility

### 2. Data Preprocessing

* Removed missing values
* Cleaned sentiment labels (lowercase + stripped spaces)
* Converted sentiment into binary values:

  * Positive → 1
  * Negative → 0

### 3. Feature Engineering

* Converted text into numerical form using **TF-IDF Vectorization**

### 4. Model Building

* Applied Logistic Regression from scikit-learn
* Split data into training and testing sets

### 5. Evaluation

Model performance was evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report
* ROC Curve (AUC Score)

---

## 📈 Output

* Model evaluation printed in terminal
* ROC curve saved in:

```
outputs/roc_curve.png
```

---

## 🛠️ Tech Stack

* Python
* Pandas
* Scikit-learn
* Matplotlib

---

## ▶️ How to Run

```bash
cd task3-logistic-regression
python src/logistic_regression.py
```

---

## 📌 Key Learning

* Difference between regression and classification
* Handling real-world text data
* Importance of data cleaning
* Feature extraction using TF-IDF
* Model evaluation techniques

---

## 🚀 Future Improvements

* Include Neutral class (multi-class classification)
* Use advanced models like Random Forest or Naive Bayes
* Improve text preprocessing (stopwords removal, stemming)
