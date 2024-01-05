import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.svm import SVR

data = pd.read_csv("D:/data/house_price_fixed.csv", index_col=None)
#PCA
pca_data = data[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']]
pca = PCA(n_components=2)
trans_data = pca.fit_transform(pca_data)

#Before SVM
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(trans_data[:,0], trans_data[:,1], data['Price'], c='r')
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("Price")
ax.set_title("Data Before SVM")
plt.show()

#Split data
X, y = np.array(np.array(trans_data)), np.array(data['Price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Built model
model = SVR(kernel='linear', C=1000, gamma=0.001)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'MSE:{mean_squared_error(y_test, y_pred)}')
print(f"R2 Score: {r2_score(y_test, y_pred)}")
# Create a meshgrid for the hyperplane
x_plane = np.linspace(trans_data[:, 0].min(), trans_data[:, 0].max(), 100)
y_plane = np.linspace(trans_data[:, 1].min(), trans_data[:, 1].max(), 100)
x_plane, y_plane = np.meshgrid(x_plane, y_plane)

# Predictions for the hyperplane
z_plane = model.predict(np.c_[x_plane.ravel(), y_plane.ravel()])
z_plane = z_plane.reshape(x_plane.shape)


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:,0], X_test[:,1], y_test, c=y_test)
ax.scatter(757, 5, model.predict([[757, 5]]), marker='*', color='red', s=200, label='Predict Data')
# Plot the hyperplane
ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.7, color='gray')
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("Price")
ax.set_title("Data after SVM")
plt.legend()
plt.show()

