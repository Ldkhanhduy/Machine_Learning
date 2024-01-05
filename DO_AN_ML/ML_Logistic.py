import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize



data = pd.read_csv("D:/data/house_price_fixed.csv", index_col=None)
np.set_printoptions(threshold=np.inf)
#Discretization
cond = [6000, 300000, 380000, 450000]
label = [0, 1, 2]
data['Price'] = pd.cut(data['Price'], bins=cond, labels=label)
#PCA
pca_data = data[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']]
pca = PCA(n_components=2)
trans_data = pca.fit_transform(pca_data)
#Split data
X, y = np.array(np.array(trans_data)), np.array(data['Price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'Accuracy:{accuracy_score(y_test, y_pred)}')
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"MSE score: {mean_squared_error(y_test, y_pred)}")


#Visualize
matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('House Price Level Prediction')
plt.show()


#ROC curve
# Dự đoán xác suất thuộc vào từng lớp
y_score = model.predict_proba(X_test)

# Chuyển đổi y_test thành one-hot encoding
y_test_bin = label_binarize(y_test, classes=model.classes_)

# Vẽ đường cong ROC
plt.figure(figsize=(8, 6))

for i in range(model.classes_.shape[0]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {model.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - One-vs-Rest (OvR)')
plt.legend()
plt.show()