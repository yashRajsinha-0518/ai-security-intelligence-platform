import shap
import pandas as pd
import numpy as np

class ShapExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        # For tree-based models (XGBoost, RF), use TreeExplainer
        try:
            self.explainer = shap.TreeExplainer(model)
            self.is_tree = True
        except Exception:
            # Fallback if not a tree model
            self.explainer = shap.Explainer(model)
            self.is_tree = False
            
    def explain_instance(self, X_scaled: np.ndarray, top_k=5):
        """
        Calculates SHAP values for a single instance.
        X_scaled should be a 2D numpy array with shape (1, n_features).
        Returns top underlying features driving the anomaly.
        """
        shap_values = self.explainer.shap_values(X_scaled)
        
        # If output is list (e.g., classification), get values for the positive class (attack)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
            
        feature_impacts = []
        for i, val in enumerate(sv):
            feature_impacts.append({
                "feature": self.feature_names[i],
                "shap_value": float(val),
                "importance": abs(float(val))
            })
            
        # Sort by absolute importance descending
        feature_impacts.sort(key=lambda x: x["importance"], reverse=True)
        return feature_impacts[:top_k]
