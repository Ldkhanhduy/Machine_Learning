import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns

#Read data
doc = pd.read_csv("D:/data/heart_fixed.csv", index_col=None)

#Create features and target
features = doc[['cp','blood_press','chol']]
target = doc['target']

#Split data
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.2, random_state=42)

#Build model
model = LogisticRegression()
model.fit(X_train, y_train)

#Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print(f'Accuracy:{accuracy}')


#Visualize with matrixplot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Heart Disease Prediction')
plt.show()
