import pandas as pd
import numpy as np

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds meaningful network traffic features to improve model performance.
    '''
    df = df.copy()
    
    # 1. Bytes per Packet Ratios
    # Add a small epsilon to avoid division by zero
    df['sbytes_per_spkt'] = df['sbytes'] / (df['spkts'] + 1e-9)
    df['dbytes_per_dpkt'] = df['dbytes'] / (df['dpkts'] + 1e-9)
    
    # 2. Total Bytes and Packets
    df['total_bytes'] = df['sbytes'] + df['dbytes']
    df['total_pkts'] = df['spkts'] + df['dpkts']
    
    # 3. Load Ratios
    df['sload_to_dload_ratio'] = df['sload'] / (df['dload'] + 1e-9)
    
    # 4. Log Transformations for heavily skewed features
    skewed_cols = ['dur', 'sbytes', 'dbytes', 'sload', 'dload', 'spkts', 'dpkts']
    for col in skewed_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
            
    # Fill NaN only for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
            
    return df
