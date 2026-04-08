import numpy as np
import pandas as pd

# load the dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.head()

df["bmi"] = df["bmi"].fillna(df["bmi"].mean())