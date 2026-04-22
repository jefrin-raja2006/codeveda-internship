# Task 2: Simple Linear Regression (Stock Prices Dataset)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/stock_prices.csv", sep="\t")

print("\nDataset loaded successfully!")
print("Columns:", df.columns)
print(df.head())


# -----------------------------
# 2. Select Features
# -----------------------------
# Predict close price using open price
df = df[['open', 'close']].dropna()

X = df[['open']]
y = df['close']


# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 4. Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# -----------------------------
# 5. Predictions
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# 6. Evaluation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# -----------------------------
# 7. Interpretation
# -----------------------------
print("\n--- Model Interpretation ---")
print(f"Slope (Coefficient): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"For every 1 unit increase in OPEN price, CLOSE price changes by {model.coef_[0]:.4f}")


# -----------------------------
# 8. Insight
# -----------------------------
print("\n--- Insight ---")
if r2 > 0.8:
    print("Strong linear relationship between open and close prices.")
elif r2 > 0.5:
    print("Moderate relationship between open and close prices.")
else:
    print("Weak relationship — linear model may not be sufficient.")


# -----------------------------
# 9. Visualization
# -----------------------------
plt.figure()

plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")

plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.title("Stock Price Prediction using Linear Regression")

plt.legend()

plt.savefig("outputs/stock_regression.png")
plt.show()