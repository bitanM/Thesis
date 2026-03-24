import pandas as pd

nodes = pd.read_csv('nodes.csv')
print("Shape:", nodes.shape)
print("\nColumns:", nodes.columns.tolist())
print("\nSample rows:")
print(nodes.head(10).to_string())
print("\nNull counts:")
print(nodes.isnull().sum())
print("\nNumeric columns stats:")
print(nodes.describe())