import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

doc = pd.read_csv("C:/Users/ACER/Downloads/winequality-red.csv")
print("Columns:",doc.columns)


f = ['alcohol','pH']
features = doc[f]
#du dung Kmeans de phan cum du lieu
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(features)

# lay cac diem trong tam lam cum
centroids = kmeans.cluster_centers_
# Lấy nhãn của từng điểm dữ liệu
labels = kmeans.labels_
# Hiển thị dữ liệu và các trọng tâm cụm trên đồ thị
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('alcohol')
plt.ylabel('pH')
plt.legend()
plt.show()
