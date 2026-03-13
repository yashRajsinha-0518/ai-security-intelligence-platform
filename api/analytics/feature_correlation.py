import pandas as pd
import numpy as np

def calculate_correlation_matrix(df: pd.DataFrame, method='pearson') -> dict:
    """
    Calculates the correlation matrix for numerical features.
    Returns a dictionary suitable for JSON serialization and Plotly Heatmaps.
    """
    # Only use numerical columns
    num_df = df.select_dtypes(include=[np.number])
    
    if num_df.empty:
        return {"features": [], "matrix": []}
        
    # Calculate correlation (fillna with 0 for safety)
    corr_matrix = num_df.corr(method=method).fillna(0)
    
    return {
        "features": corr_matrix.columns.tolist(),
        "matrix": corr_matrix.values.tolist()
    }
