import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("anime-filtered.csv")

print(df.head())
print(df.describe())
print(df.info())

df['Score'].hist(bins=50)
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Distribution of Anime Scores')
plt.show()