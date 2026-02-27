# ==============================
# 📌 Import Libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# ==============================
# 📌 Load Dataset
# ==============================
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")


# ==============================
# 📌 Feature Engineering
# ==============================

# Convert dates
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

# Extract date features
df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month
df["Ship_Year"] = df["Ship Date"].dt.year
df["Ship_Month"] = df["Ship Date"].dt.month

# Shipping duration
df["Shipping_Days"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

# Drop original dates
df.drop(["Order Date", "Ship Date"], axis=1, inplace=True)


# ==============================
# 📌 Business Features
# ==============================
df["Sales_per_Quantity"] = df["Sales"] / df["Quantity"]
df["Discount_Impact"] = df["Sales"] * df["Discount"]


# ==============================
# 📌 Remove Unnecessary Columns
# ==============================
df.drop([
    "Row ID",
    "Order ID",
    "Customer ID",
    "Customer Name",
    "Product ID",
    "Product Name"
], axis=1, inplace=True)


# ==============================
# 📌 Encode Categorical Variables
# ==============================
df = pd.get_dummies(df, drop_first=True)


# ==============================
# 📌 Handle Outliers (Profit)
# ==============================
q1 = df["Profit"].quantile(0.25)
q3 = df["Profit"].quantile(0.75)
iqr = q3 - q1

df = df[
    (df["Profit"] >= q1 - 1.5 * iqr) &
    (df["Profit"] <= q3 + 1.5 * iqr)
]


# ==============================
# 📌 Define Features & Target
# ==============================
X = df.drop("Profit", axis=1)
y = df["Profit"]

# ✅ Save feature names BEFORE scaling
feature_names = X.columns


# ==============================
# 📌 Feature Scaling
# ==============================
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ==============================
# 📌 Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 📌 Model Training
# ==============================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)


# ==============================
# 📌 Prediction & Evaluation
# ==============================
y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# ==============================
# 📌 Feature Importance Visualization
# ==============================
# ==============================
# 📌 Feature Importance (Clean Plot)
# ==============================

importance = model.feature_importances_

feat_imp = pd.Series(importance, index=feature_names)

# ✅ Select top 15 important features
top_features = feat_imp.sort_values(ascending=False).head(15)

plt.figure(figsize=(10,6))
top_features.sort_values().plot(kind="barh")

plt.title("Top 15 Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()

# ==============================
# 📌 Save Model
# ==============================
import joblib

# Save model
joblib.dump(model, "profit_prediction_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save feature names
joblib.dump(feature_names, "features.pkl")

print("✅ Model saved successfully!")