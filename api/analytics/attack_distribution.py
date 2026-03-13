import pandas as pd

def calculate_attack_distribution(df: pd.DataFrame, preds: list) -> dict:
    """
    Groups predictions by protocol/service to find attack distributions.
    Expects df to have 'proto' and 'service' columns, and preds to be the model predictions.
    """
    # Create a temporary df to avoid modifying original
    temp_df = df.copy()
    temp_df['prediction'] = preds
    
    distributions = {
        "overall": {},
        "by_protocol": {},
        "by_service": {}
    }
    
    # Overall
    attacks = int(sum(preds))
    normal = int(len(preds) - attacks)
    distributions["overall"] = {"attack": attacks, "normal": normal}
    
    # By protocol (only for attacks)
    if 'proto' in temp_df.columns:
        attack_df = temp_df[temp_df['prediction'] == 1]
        proto_counts = attack_df['proto'].value_counts().to_dict()
        distributions["by_protocol"] = proto_counts
        
    # By service (only for attacks)
    if 'service' in temp_df.columns:
        attack_df = temp_df[temp_df['prediction'] == 1]
        service_counts = attack_df['service'].value_counts().to_dict()
        distributions["by_service"] = service_counts
        
    return distributions
