import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# =========================
# PATH HANDLING (PRO)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

train_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.parquet")
test_path = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.parquet")

print("Loading dataset...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

df = pd.concat([train_df, test_df], ignore_index=True)
# =========================
# CLEAN MISSING VALUES
# =========================
print("Cleaning missing values...")

df.replace("-", pd.NA, inplace=True)
df = df.apply(pd.to_numeric, errors='ignore')

# Fill numeric NaNs with median
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical NaNs with 'unknown'
for col in df.select_dtypes(include='object').columns:
    df[col].fillna("unknown", inplace=True)

print("Total rows:", df.shape)

# =========================
# CREATE BINARY LABEL
# =========================
# UNSW label column = 'label' (0 normal, 1 attack)
y = df['label']

X = df.drop(columns=['label'])

# =========================
# HANDLE CATEGORICAL COLS
# =========================
print("Encoding categorical columns...")

# =========================
# FORCE FULL NUMERIC CONVERSION (ROBUST)
# =========================
print("Encoding categorical columns (robust)...")

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    else:
        # Try converting hidden mixed columns
        try:
            X[col] = pd.to_numeric(X[col])
        except:
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# MODEL TRAINING
# =========================
print("Training XGBoost model...")

model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",   # CPU optimized
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# SAVE ARTIFACTS
# =========================
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "ids_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nModel saved successfully!")