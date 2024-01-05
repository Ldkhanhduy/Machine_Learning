import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("D:/data/house_price_fixed.csv", index_col=None)

features = data[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']]
target = data['Price']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)

#built
ls = LassoCV(alphas=None, cv=5)
ls.fit(X_train_scaled, y_train)
#best alpha
print(f"Best Alpha: {ls.alpha_}")

#Predict
y_pred = ls.predict(X_test_scaled)
#performance
print(F"R2 Score: {r2_score(y_test, y_pred)}")
print(f"MSE Score: {mean_squared_error(y_test, y_pred)}")

plt.figure(figsize=(10, 6))
plt.plot(ls.coef_, marker='o', linestyle='None', color='r', label='Lasso coefficients')
plt.axhline(0, linestyle='--', color='gray', linewidth=0.5, label='Zero coefficient line')
plt.title('Lasso Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.legend()
plt.show()
#Visualize
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.xlabel('Thực tế')
plt.ylabel('Dự đoán')
plt.title('Biểu đồ scatter giữa giá trị thực tế và dự đoán')
plt.show()