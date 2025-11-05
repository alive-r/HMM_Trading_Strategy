import pandas as pd

df = pd.read_parquet("data_cache/index_features.parquet")
print(df.head())       
print(df.info())      
print(df.describe())   