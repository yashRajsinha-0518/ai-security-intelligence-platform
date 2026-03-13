import pandas as pd
import numpy as np

def generate_dataset_health(df: pd.DataFrame) -> dict:
    """
    Analyzes the dataframe for basic health metrics: missing values,
    cardinality, data types, and overall shape.
    """
    total_rows = len(df)
    missing_values = df.isnull().sum().to_dict()
    missing_percent = (df.isnull().sum() / total_rows * 100).to_dict()
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    health_report = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "numerical_features_count": len(numerical_cols),
        "categorical_features_count": len(categorical_cols),
        "missing_values": missing_values,
        "missing_percentage": missing_percent
    }
    
    return health_report
