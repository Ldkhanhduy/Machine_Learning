import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal


#Read data
data = pd.read_csv("D:/data/heart.csv", index_col=None)
#Collect features and target
X = np.array(data[['blood_press', 'chol']]) #Features
y = np.array(data[['target']]) #Target
#Visualize data for model's decision
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, marker='o', edgecolors='k')
plt.xlabel("Blood pressure", fontweight='bold')
plt.ylabel("Cholestarol", fontweight='bold')
plt.title("Scatter plot for blood pressure and cholestarol", fontweight='bold', size=15)
plt.legend()
plt.show()
#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11)
#Test model with best Gamma and C
# param_distribs = {'C': reciprocal(20, 200000), 'gamma': reciprocal(0.0001, 0.1)}
# random_search = RandomizedSearchCV(svm.SVC(kernel='rbf'), param_distributions=param_distribs, n_iter=10, cv=5)
# random_search.fit(X_train, y_train)
# best_params = random_search.best_params_
# print(f'Model show best performance with:{best_params}')
#Built SVM model
model = svm.SVC(kernel='rbf', gamma=0.07, C=729)
model.fit(X_train,y_train)
#Kiểm tra hiệu suất mô hình
performance = model.score(X_test, y_test)
print(f'Performance:{performance}')
#predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print(f'Accuracy:{accuracy}')
#Visualize with 3D plot
Z = model.decision_function(np.c_[X_test[:, 0], X_test[:, 1]])
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 50),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 50))

zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
zz = zz.reshape(xx.shape)
ax.plot_surface(xx, yy, zz, alpha=0.3, cmap=plt.cm.coolwarm)
# plt.contour(xx, yy, zz, levels=[0], linewidths=2, colors='black')
ax.scatter(X_test[:,0], X_test[:,1], Z, c=y_test, cmap=plt.cm.coolwarm, marker='o', edgecolors='k')
ax.set_xlabel("Blood pressure", fontweight='bold')
ax.set_ylabel("Cholestarol", fontweight='bold')
ax.set_zlabel("Kernal value", fontweight='bold')
plt.title("SVM Classifier With RBF Kernel (3D)", size=20, fontweight='bold')
plt.show()

