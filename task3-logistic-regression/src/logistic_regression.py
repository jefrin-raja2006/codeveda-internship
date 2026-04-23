# Task 3: Logistic Regression (Sentiment Classification)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/dataset.csv", sep=None, engine='python')

print("\nDataset Loaded!")
print(df.head())
print("\nColumns:", df.columns)


# -----------------------------
# 2. Preprocessing (FIXED)
# -----------------------------
# Keep only needed columns
df = df[['Text', 'Sentiment']].dropna()

# Convert to string (safe)
df['Text'] = df['Text'].astype(str)
df['Sentiment'] = df['Sentiment'].astype(str)

# Clean sentiment column
df['Sentiment'] = df['Sentiment'].str.strip().str.lower()

# Debug check
print("\nUnique Sentiments:", df['Sentiment'].unique())

# Keep only positive & negative
df = df[df['Sentiment'].isin(['positive', 'negative'])]

# Convert to binary
df['Sentiment'] = df['Sentiment'].map({
    'positive': 1,
    'negative': 0
})

# Check distribution
print("\nClass Distribution:\n", df['Sentiment'].value_counts())


# -----------------------------
# 3. Text → Numeric (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['Text'])
y = df['Sentiment']


# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 5. Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")


# -----------------------------
# 6. Predictions
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# 7. Evaluation
# -----------------------------
print("\n--- Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# -----------------------------
# 8. ROC Curve
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig("outputs/roc_curve.png")
plt.show()