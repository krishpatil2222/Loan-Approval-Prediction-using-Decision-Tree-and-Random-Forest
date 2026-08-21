# Loan Approval Prediction using Decision Tree and Random Forest

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
dataset = pd.read_csv("loan_approval_dataset.csv")

# Remove extra spaces from column names
dataset.columns = dataset.columns.str.strip()

print("First 5 Records")
print(dataset.head())

print("\nDataset Information")
print(dataset.info())

print("\nMissing Values")
print(dataset.isnull().sum())

# Step 2: Data Preprocessing

# Remove Loan ID
dataset.drop("loan_id", axis=1, inplace=True)

# Remove leading/trailing spaces from string values
for column in dataset.select_dtypes(include=["object", "string"]).columns:
    dataset[column] = dataset[column].astype(str).str.strip()

# Encode categorical columns
encoder = LabelEncoder()

categorical_columns = ["education", "self_employed", "loan_status"]

for column in categorical_columns:
    dataset[column] = encoder.fit_transform(dataset[column])

print("\nEncoded Dataset")
print(dataset.head())

# Independent and Dependent Variables
X = dataset.drop("loan_status", axis=1)
Y = dataset["loan_status"]

# Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Step 3: Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, Y_train)

# Step 4: Decision Tree Prediction
dt_prediction = dt_model.predict(X_test)

dt_accuracy = accuracy_score(Y_test, dt_prediction)

print("\nDecision Tree Accuracy: {:.2f}%".format(dt_accuracy * 100))

# Step 5: Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, Y_train)

# Step 6: Random Forest Prediction
rf_prediction = rf_model.predict(X_test)

rf_accuracy = accuracy_score(Y_test, rf_prediction)

print("Random Forest Accuracy: {:.2f}%".format(rf_accuracy * 100))

# Step 7: Compare Models
print("\nModel Comparison")

if rf_accuracy > dt_accuracy:
    print("Random Forest performs better than Decision Tree.")
elif rf_accuracy < dt_accuracy:
    print("Decision Tree performs better than Random Forest.")
else:
    print("Both models have the same accuracy.")

# Step 8: Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(importance["Feature"], importance["Importance"])

plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()