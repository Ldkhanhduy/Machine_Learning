import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns


#Read data
doc = pd.read_csv("D:/data/heart_fixed.csv")

#Choose features and target
features = doc[['blood_press','chol','max_heart_rate']]
target = doc['age']

#Split data to train and test data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=15)

#Build model
linear = LinearRegression()
linear.fit(X_train, y_train)

#Predict
y_pred = linear.predict(X_test)

#Result
print(f'Mean Absolute Error:{mean_absolute_error(y_test,y_pred)}')
print(f'Mean Squared Error:{mean_squared_error(y_test, y_pred)}')
print(f'R-Squared:{r2_score(y_test, y_pred)}')

#Visualize result
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', alpha=0.5)
#vẽ một đường thẳng từ giá trị thấp nhất đến cao nhất trên hai trục x và y
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=2)
plt.title('Biểu đồ mô hình hồi quy tuyến tính đa biến')
plt.xlabel('Giá trị thực tế')
plt.ylabel(' Giá trị dự đoán')
plt.grid(True)
plt.show()

residuals = y_test - y_pred

# Vẽ biểu đồ residuals
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Biểu đồ Residuals')
plt.xlabel('Giá trị Thực tế')
plt.ylabel('Residuals')
plt.show()