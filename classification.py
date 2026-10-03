import pandas as pd

print("Loading data...")

orders = pd.read_csv("data/orders.csv")
order_products = pd.read_csv("data/order_products__prior.csv")

print("Merging datasets...")

df = order_products.merge(
    orders[
        [
            "order_id",
            "user_id",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order"
        ]
    ],
    on="order_id",
    how="left"
)

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nSAMPLE:")
print(df.head())

print("\nCLASS DISTRIBUTION:")
print(df["reordered"].value_counts())

print("\nCLASS PERCENTAGES:")
print(df["reordered"].value_counts(normalize=True) * 100)

# -------------------------------------------------
# CLASSIFICATION MODEL - DECISION TREE
# -------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("\nCreating sample for model training...")

# Stratified sample to preserve class distribution
df_sample = (
    df.groupby("reordered", group_keys=False)
      .sample(n=200000, random_state=42)
)

features = [
    "order_number",
    "order_dow",
    "order_hour_of_day",
    "days_since_prior_order",
    "add_to_cart_order"
]

X = df_sample[features].copy()
y = df_sample["reordered"]

# Fill missing values
X["days_since_prior_order"] = X["days_since_prior_order"].fillna(0)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Create model
model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

print("\nTraining Decision Tree...")

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nDECISION TREE RESULTS")
print("---------------------")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------
# CLASSIFICATION MODEL - RANDOM FOREST
# -------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRANDOM FOREST RESULTS")
print("---------------------")
print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# -------------------------------------------------
# CLASSIFICATION MODEL - NAIVE BAYES
# -------------------------------------------------

from sklearn.naive_bayes import GaussianNB

print("\nTraining Naive Bayes...")

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\nNAIVE BAYES RESULTS")
print("-------------------")
print("Accuracy :", accuracy_score(y_test, nb_pred))
print("Precision:", precision_score(y_test, nb_pred))
print("Recall   :", recall_score(y_test, nb_pred))
print("F1 Score :", f1_score(y_test, nb_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, nb_pred))

# -------------------------------------------------
# MODEL COMPARISON CHART
# -------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

models = ["Decision Tree", "Random Forest", "Naive Bayes"]

accuracy = [
    accuracy_score(y_test, y_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, nb_pred)
]

precision = [
    precision_score(y_test, y_pred),
    precision_score(y_test, rf_pred),
    precision_score(y_test, nb_pred)
]

recall = [
    recall_score(y_test, y_pred),
    recall_score(y_test, rf_pred),
    recall_score(y_test, nb_pred)
]

f1 = [
    f1_score(y_test, y_pred),
    f1_score(y_test, rf_pred),
    f1_score(y_test, nb_pred)
]

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10, 6))

plt.bar(x - 1.5*width, accuracy, width, label="Accuracy")
plt.bar(x - 0.5*width, precision, width, label="Precision")
plt.bar(x + 0.5*width, recall, width, label="Recall")
plt.bar(x + 1.5*width, f1, width, label="F1 Score")

plt.xlabel("Classification Algorithm")
plt.ylabel("Score")
plt.title("Classification Algorithm Performance Comparison")

plt.xticks(x, models)
plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)

plt.show()