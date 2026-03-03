import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

train_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.parquet")
test_path = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.parquet")

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nColumns sample:", train.columns[:10])
print("\nFirst rows:")
print(train.head())