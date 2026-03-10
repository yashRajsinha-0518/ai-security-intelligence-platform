import pandas as pd
import numpy as np
import os
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from feature_engineer import add_engineered_features
from evaluation import evaluate_model
from experiment_tracker import ExperimentTracker

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure models directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

train_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.parquet")
test_path = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.parquet")

def load_data():
    print("Loading datasets...")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    return train, test

def prepare_pipeline(X_train):
    # Identify categorical and numerical columns
    # We ignore the 'id' and 'attack_cat' for the binary classification model
    drop_cols = ['id', 'attack_cat'] if 'id' in X_train.columns else ['attack_cat']
    drop_cols = [c for c in drop_cols if c in X_train.columns]
    
    categorical_cols = X_train.drop(columns=drop_cols).select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.drop(columns=drop_cols).select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor, numerical_cols, categorical_cols

def train_and_evaluate():
    train, test = load_data()
    
    # Feature Engineering
    print("Applying feature engineering...")
    train = add_engineered_features(train)
    test = add_engineered_features(test)
    
    y_train = train['label']
    y_test = test['label']
    
    drop_cols = ['label', 'id', 'attack_cat']
    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
    
    preprocessor, num_cols, cat_cols = prepare_pipeline(X_train)
    
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="logloss"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
    }
    
    tracker = ExperimentTracker(os.path.join(BASE_DIR, "mlflow_tracking"))
    best_model_name = None
    best_f1 = -1
    best_pipeline = None

    # Save feature columns
    feature_columns = X_train.columns.tolist()
    with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_columns, f)

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("classifier",clf)])
        pipeline.fit(X_train, y_train)
        
        # Eval
        print(f"Evaluating {name}...")
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        
        metrics, cm = evaluate_model(y_test, y_pred, y_prob, name)
        
        # Track
        params = {"model": name}
        tracker.log_experiment_json(name, params, metrics)
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest Model: {best_model_name} with F1: {best_f1:.4f}")
    
    # Save the pipeline
    with open(os.path.join(MODEL_DIR, "ids_pipeline.pkl"), "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"[{best_model_name}] Pipeline saved to {MODEL_DIR}/ids_pipeline.pkl successfully!")

if __name__ == "__main__":
    train_and_evaluate()
