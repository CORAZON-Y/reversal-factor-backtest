import pandas as pd
import os

# 获取脚本所在目录，然后找到数据文件
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "basic_data", "daily_data.parquet")

df = pd.read_parquet(data_path)

print(df.head())
print(df.columns)
print(df.shape)