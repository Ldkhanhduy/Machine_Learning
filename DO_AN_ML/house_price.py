import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


#read data and check missing value
data = pd.read_csv("D:/data/house_price.csv", index_col=None)
data.info()

#check noise value
columns_to_check = data.columns
for i in columns_to_check:
    # Transforme all columns to numberic
    numeric_values = pd.to_numeric(data[i], errors='coerce')

    # Check unsuccess value
    non_numeric_values = data[i][numeric_values.isna()]

    if non_numeric_values.empty:
        print(f"Column '{i}' contain numberic only.")
    else:
        print(f"Column '{i}' has non numberic value: {len(non_numeric_values)}")

#Check duplicate
duplicate = data.duplicated()
duplicates_data = pd.DataFrame({
'Row': range(1, len(data) + 1),
'Duplicated': duplicate
})
duplicates_data = duplicates_data[duplicates_data['Duplicated']]
#Result for duplicate
print(f'Sum of duplicate:{len(duplicates_data)}')

#find outliers
plt.boxplot(data['Price'], vert=False)
plt.title("Boxplot find outliers for columns 'Price'")
plt.xlabel("Price")
plt.show()
def find_outliers(data):
    global q_list
    q_list = []
    sorted_data = data.sort_values()
    for q, p in {"Q1": 25, "Q2": 50, "Q3": 75}.items():
    # Calculate Q1, Q2, Q3 and IQR.
        Q = np.percentile(sorted_data, p, interpolation = 'midpoint')
        q_list.append(Q)
        print("Checking...", q)
        print("{}: {} percentile of the {} values is, ".format(q,p,data.name), Q)
    global Q1, Q2, Q3
    Q1 = q_list[0]
    Q2 = q_list[1]
    Q3 = q_list[2]
    IQR = Q3 - Q1
    print("Interquartile range is", IQR)
    # Find the lower and upper limits as Q1 – 1.5 IQR and Q3 + 1.5 IQR, respectively
    global low_lim, up_lim
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    print(" ")
    print("Checking limits")
    print("low_limit is", low_lim)
    print("up_limit is", up_lim)
    # Find outliers in the dataset
    outliers =[]
    for x in sorted_data:
        if ((x> up_lim) or (x<low_lim)):
            outliers.append(x)

    print("\nOutliers are being added to list. Please wait!")
    print("\nOutliers in the dataset is", outliers)

find_outliers(data['Price'])
#clean outliers
data = data[(data['Price'] < up_lim) & (data['Price'] >
low_lim)]
data.info()

#change "Neighborhood"
data['Neighborhood'] = data['Neighborhood'].replace({'Rural':0, 'Urban':1, 'Suburb':2})

data.to_csv("D:/data/house_price_fixed.csv", index=False)




