import pandas as pd
df = pd.read_parquet('data/UNSW_NB15_testing-set.parquet')
df = df.drop(columns=['id', 'label'], errors='ignore')
attack_row = df[df['attack_cat'] != 'Normal'].iloc[5].values
normal_row = df[df['attack_cat'] == 'Normal'].iloc[5].values

with open("sample_test_runs.md", "w") as f:
    f.write("**Attack Traffic Example (`Reconnaissance` or `DoS` etc.):**\n\n")
    f.write(",".join(map(str, attack_row)))
    f.write("\n\n**Normal Traffic Example:**\n\n")
    f.write(",".join(map(str, normal_row)))
