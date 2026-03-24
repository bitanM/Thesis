import pandas as pd

edges = pd.read_csv('edges_pmi_pruned_org.csv')
print("Shape:", edges.shape)
print("\nColumns:", edges.columns.tolist())
print("\nSample rows:")
print(edges.head(5).to_string())
print("\nWeight stats:")
print(edges['weight'].describe())
print("\nNull counts:")
print(edges.isnull().sum())