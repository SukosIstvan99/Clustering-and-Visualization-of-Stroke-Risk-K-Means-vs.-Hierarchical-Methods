import pandas as pd 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage 

data_path = "healthcare-dataset-stroke-data.csv"
df = pd.read_csv(data_path)

df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

features = ['age', 'avg_glucose_level', 'bmi']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(12,8))
dendrogram(
    linked,
    orientation='top',
    labels=df.index.tolist(),
    distance_sort='descending',
    show_leaf_counts=True
)

plt.title('Hierarchikus klaszterek')
plt.xlabel('Mintak')
plt.ylabel('Tavolsag')
plt.show()