import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "healthcare-dataset-stroke-data.csv"
df = pd.read_csv(data_path)

df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

features = ['age', 'avg_glucose_level', 'bmi']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

def categorize_age(age):
    if age <= 20:
        return '0-20'
    elif 21 <= age <= 40:
        return '21-40'
    elif 41 <= age <= 60:
        return '41-60'
    elif 61 <= age <= 80:
        return '61-80'
    else:
        return '80+'

df['age_category'] = df['age'].apply(categorize_age)

print("Stroke esély életkori kategóriánként (futás közben):\n")
age_groups = df.groupby('age_category')

for age_category, group in age_groups:
    stroke_rate = group['stroke'].mean() * 100  # százalékos arány
    print(f"Korosztály: {age_category}, Stroke esély: {stroke_rate:.2f}%")

plt.figure(figsize=(10, 6))
sns.barplot(x='age_category', y='stroke', data=df, estimator=lambda x: sum(x)/len(x)*100)
plt.title('Stroke Valószínűség Életkori Kategóriánként')
plt.xlabel('Életkori Kategóriák')
plt.ylabel('Stroke Valószínűség (%)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['stroke'], palette='coolwarm', s=50
)
plt.title('Stroke Distribution within Clusters')
plt.xlabel('Age (scaled)')
plt.ylabel('Avg Glucose Level (scaled)')
plt.legend(title="Stroke (0 = No, 1 = Yes)")
plt.show()
