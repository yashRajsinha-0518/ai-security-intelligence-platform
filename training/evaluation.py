from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model"):
    '''
    Computes standard evaluation metrics.
    '''
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    # Calculate False Positive Rate
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-9)
    
    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
            
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positive_rate": fpr,
    }
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
        
    print(f"--- Evaluation for {model_name} ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return metrics, cm
