import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("=== Code Started ===")

# Load dataset
df = pd.read_csv("data/iris.csv")

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Dataset Info ===")
print(df.info())

# Check missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Split features and target
X = df.drop("species", axis=1)
y = df["species"]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

print("\nEncoded Classes:", le.classes_)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# Save processed data
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df["target"] = y

processed_df.to_csv("processed_iris.csv", index=False)

print("\n=== Processed file saved ===")