import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r'Measures/data/public/california_housing.csv')
print(df.shape)
# Benchmark run artifact
