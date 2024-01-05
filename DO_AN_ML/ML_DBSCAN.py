import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns

data = pd.read_csv("D:/data/air_fixed.csv", index_col=None)
# Tính toán khoảng cách
k = 4  # số điểm gần nhất
neigh = NearestNeighbors(n_neighbors=k)
distances, _ = neigh.fit(data).kneighbors(data)

# Sắp xếp và lấy giá trị khoảng cách
sorted_distances = np.sort(distances[:, -1])

# Vẽ biểu đồ k-distance
plt.plot(sorted_distances)
plt.xlabel('Data Points')
plt.ylabel(f'{k}-Distance')
plt.title(f'{k}-Distance Plot')
plt.show()

#all eps and min sample test
param_dist = {
    'eps': uniform(0.01, 0.05),
    'min_samples': randint(2, 10)
}
#find best eps and min sample for model
dbs = DBSCAN()
random_search = RandomizedSearchCV(dbs, param_distributions=param_dist,
                                   n_iter=10, scoring='roc_auc', cv=5, random_state=42)
random_search.fit(data)

#show best performance
print(f"Best paramenter: {random_search.best_params_}")

#built model
model = DBSCAN(eps=0.029, min_samples=6)
model.fit(data)

#performance
score = silhouette_score(data, model.labels_)
print(f"Silhouette Score: {score}")
print(f"Davies Bouldin Score: {davies_bouldin_score(data, model.labels_)}")

#visualize
#before
sns.pairplot(data)
plt.show()

#after
data['Cluster'] = model.labels_
sns.pairplot(data, hue='Cluster', palette='coolwarm')
plt.show()
