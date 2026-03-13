import pandas as pd
import numpy as np

def detect_outliers_iqr(df: pd.DataFrame, features: list = None) -> dict:
    """
    Calculates Interquartile Range (IQR) for specified features to detect statistical outliers.
    If features is None, defaults to prominent numerical features if present.
    """
    default_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'rate']
    features_to_check = features if features else [f for f in default_features if f in df.columns]
    
    outliers_report = {}
    
    for feature in features_to_check:
        if feature not in df.columns or not pd.api.types.is_numeric_dtype(df[feature]):
            continue
            
        series = df[feature].dropna()
        if len(series) == 0:
            continue
            
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        outliers_report[feature] = {
            "q1": float(Q1),
            "q3": float(Q3),
            "iqr": float(IQR),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(len(outliers)),
            "outlier_percentage": float(len(outliers) / len(series) * 100) if len(series) > 0 else 0
        }
        
    return outliers_report

def calculate_statistical_moments(df: pd.DataFrame, features: list = None) -> dict:
    """
    Calculates mean, variance, skewness, and kurtosis.
    """
    default_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'rate']
    features_to_check = features if features else [f for f in default_features if f in df.columns]
    
    moments_report = {}
    
    for feature in features_to_check:
         if feature not in df.columns or not pd.api.types.is_numeric_dtype(df[feature]):
            continue
            
         series = df[feature].dropna()
         if len(series) == 0:
             continue
             
         moments_report[feature] = {
             "mean": float(series.mean()),
             "variance": float(series.var()) if len(series) > 1 else 0,
             "skewness": float(series.skew()) if len(series) > 2 else 0,
             "kurtosis": float(series.kurtosis()) if len(series) > 3 else 0,
             "min": float(series.min()),
             "max": float(series.max())
         }
         
    return moments_report
