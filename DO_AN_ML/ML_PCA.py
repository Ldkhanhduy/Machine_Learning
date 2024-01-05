import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

data = pd.read_csv("D:/data/air_fixed.csv", index_col=None)

#t-SNE
tsne = TSNE(n_components=2, random_state=42)
trans_data = tsne.fit_transform(data)
#before
plt.scatter(trans_data[:, 0], trans_data[:, 1])
plt.title('Data Before Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

#built model
hier = AgglomerativeClustering(n_clusters=3)
hier_labels = hier.fit_predict(trans_data)
#performance
print(f"Davies Bouldin Score: {davies_bouldin_score(trans_data, hier_labels)}")
print(f"Silhouette Score: {silhouette_score(trans_data, hier_labels)}")

#visualize
#dondrogram
linkage_matrix = linkage(trans_data, method='ward')
dendrogram(linkage_matrix)
plt.title("Dendrogram")
plt.xlabel("Data Point")
plt.ylabel("Distance")
plt.show()

#scatter
plt.scatter(trans_data[:, 0], trans_data[:, 1], c=hier_labels, cmap='coolwarm')
plt.title('Mean Shift Clustering using t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
