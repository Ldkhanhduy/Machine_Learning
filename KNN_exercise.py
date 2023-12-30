import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

df = pd.read_csv('D:/data/heart_fixed.csv', index_col=None)
# Sao lưu DataFrame để tránh mất mát thông tin
data = df.copy()

selected_columns = ['blood_press', 'max_heart_rate', 'chol']
data = data[selected_columns]

X = data[['blood_press','max_heart_rate']]
y = data['chol']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


best_k = None
best_mse = float('inf')

for k in range(1, 21):
    knn_model = KNeighborsRegressor(n_neighbors=k)
    mse_scores = -cross_val_score(knn_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    avg_mse = np.mean(mse_scores)

    if avg_mse < best_mse:
        best_mse = avg_mse
        best_k = k

print(f'Best k: {best_k}')

knn_model = KNeighborsRegressor(n_neighbors=best_k)

knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
# Đánh giá mô hình bằng mean squared error
mse = mean_squared_error(y_test.astype(int), np.round(y_pred).astype(int))
print(f'Mean Squared Error: {mse}')

# Đánh giá mô hình bằng accuracy
accuracy = accuracy_score(y_test.astype(int), np.round(y_pred).astype(int))
print(f'Accuracy: {100*accuracy:.2f}%')
r2 = r2_score(y_test, y_pred)
print(f'R-squared (R²): {r2}')

# Tạo một điểm dữ liệu mới để dự đoán
new_data_point = pd.DataFrame({'blood_press': [200], 'max_heart_rate': [180]})


# Dự đoán giá trị cho điểm dữ liệu mới
predicted_chol = knn_model.predict(new_data_point)

print(f'Predicted Cholesterol for the new data point: {predicted_chol[0]}')

# Tạo một biểu đồ mới
plt.figure()

# Vẽ biểu đồ trực quan hóa kết quả với các điểm dữ liệu ban đầu
scatter = plt.scatter(X_train['blood_press'], X_train['max_heart_rate'], c=y_train, cmap='viridis', label='Predicted Data')
cbar = plt.colorbar(scatter, label='Cholesterol Concentration')
cbar.set_label('Cholesterol Concentration (mg/dL)', fontsize=15)
plt.xlabel('Blood Pressure', fontsize=20, labelpad=10)
plt.ylabel('Maximum Heart Rate', fontsize=18, labelpad=22)
plt.title('Using KNN Predicted Cholesterol Plot (k=1)', fontsize=23, fontweight='bold')

# Thêm điểm dữ liệu mới vào biểu đồ mới và phản ánh màu sắc theo dự đoán "Cholesterol Concentration"
predicted_chol_new_data = knn_model.predict(new_data_point)
color_map = plt.cm.get_cmap('viridis')
color_norm = plt.Normalize(vmin=min(y_pred), vmax=max(y_pred))
scatter_new_data = plt.scatter(new_data_point['blood_press'], new_data_point['max_heart_rate'],
                               c=predicted_chol_new_data, cmap=color_map, norm=color_norm, marker='*', s=200, label='New Data Point')

# Thêm ghi chú cho điểm mới
plt.legend()

# Hiển thị biểu đồ
plt.show()
