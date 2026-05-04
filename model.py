# Libraries

import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("dataset.csv")

# =========================
# CLEAN DATA
# =========================
# Drop duplicates
df = df.drop_duplicates()

# Handle missing values
# Separate features & target
target_col = "Has_Hypertension"  

X = df.drop(target_col, axis=1)
y = df[target_col]

# Encode target if categorical
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)
else:
    le = None

# Identify numeric & categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# =========================
# PREPROCESSING PIPELINE
# =========================
# Clean column names
df.columns = df.columns.str.strip()

# Target column
target_col = "Has_Hypertension"

X = df.drop(target_col, axis=1)
y = df[target_col]

# Encode target if needed
from sklearn.preprocessing import LabelEncoder

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

# Identify column types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Preprocessing pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# APPLY PREPROCESSING
# =========================
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# =========================
# TRAIN MODELS
# =========================
log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

print("Logistic Accuracy:", accuracy_score(y_test, log_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\nLogistic Confusion Matrix:\n", confusion_matrix(y_test, log_pred))
print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# =========================
# SAVE MODELS
# =========================
pickle.dump(preprocessor, open("pipeline.pkl", "wb"))
pickle.dump(log_model, open("log_model.pkl", "wb"))
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("✅ Models saved successfully!")

# =========================
# CASCADE PREDICTION LOGIC
# =========================
def cascaded_predict(log_model, rf_model, X, threshold=0.7):
    prob = log_model.predict_proba(X)[0][1]

    if prob >= threshold or prob <= (1 - threshold):
        return log_model.predict(X)[0], "Logistic Regression"
    else:
        return rf_model.predict(X)[0], "Random Forest"
    
    


# =========================
# EVALUATION (ROC AUC)
# =========================
log_probs = log_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]

log_auc = roc_auc_score(y_test, log_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

print(f"Logistic Regression AUC: {log_auc:.4f}")
print(f"Random Forest AUC: {rf_auc:.4f}")

# Also keeping your accuracy printouts for comparison
log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

print(f"Logistic Accuracy: {accuracy_score(y_test, log_pred):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")


def plot_roc_curves(y_true, log_p, rf_p):
    # Calculate AUCs inside the function or pass them as arguments
    from sklearn.metrics import roc_curve, auc
    
    log_auc = auc(*roc_curve(y_true, log_p)[:2])
    rf_auc = auc(*roc_curve(y_true, rf_p)[:2])

    plt.figure(figsize=(8, 6))
    
    # Logistic Regression Curve
    fpr_log, tpr_log, _ = roc_curve(y_true, log_p)
    plt.plot(fpr_log, tpr_log, label=f'Logistic (AUC = {log_auc:.2f})', color='blue')
    
    # Random Forest Curve
    fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_p)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.2f})', color='green')
    
    # Random Guess Line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show() # This will open a window with the graph
# Call the function
plot_roc_curves(y_test, log_probs, rf_probs)

# =========================
# SAVE METRICS
# =========================
# Create a dictionary of the scores
model_metrics = {
    "log_auc": log_auc,
    "rf_auc": rf_auc,
    "log_accuracy": accuracy_score(y_test, log_pred),
    "rf_accuracy": accuracy_score(y_test, rf_pred)
}

# Save using pickle
with open("model_metrics.pkl", "wb") as f:
    pickle.dump(model_metrics, f)

print("✅ AUC and Accuracy metrics saved successfully!")



    
# =========================
# CONFUSION MATRIX CHART
# =========================    
import matplotlib.pyplot as plt

def plot_conf_matrix(cm, title):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_conf_matrix(confusion_matrix(y_test, log_pred), "Logistic Regression")
plot_conf_matrix(confusion_matrix(y_test, rf_pred), "Random Forest")    
