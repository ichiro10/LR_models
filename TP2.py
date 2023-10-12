import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('c:/Users/ghamm/OneDrive/Bureau/UQAC/TP/AA/boston.csv')
print(data)
print(data.info())
print(data.shape)
print(data.nunique())
print(data.describe(include='all'))
#missed values 
print(data.isnull().sum())

#categorical data 
dummies = pd.get_dummies(data["RAD"],dtype=int ,prefix="RAD")
# Concatenate the original DataFrame with the dummy variables
data = pd.concat([data, dummies], axis=1)

# Drop the original categorical column if needed
data = data.drop(columns=['RAD'])

print(data)

#outliers 
qt = QuantileTransformer(output_distribution='normal')

for col in data.columns:
    data[col] = data[col] = qt.fit_transform(pd.DataFrame(data[col]))
'''''
for column in data.columns : 
    plt.figure(figsize = (14,4))
    sns.histplot(data[column])
    plt.title(column)
    plt.show()    

'''''

for col in data:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    whisker_width = 1.5
    lower_whisker = q1 - (whisker_width * iqr)
    upper_whisker = q3 + whisker_width * iqr
    data[col] = np.where(data[col] > upper_whisker, upper_whisker, np.where(data[col] < lower_whisker, lower_whisker, data[col]))
'''''
for column in data:
        sns.boxplot(data = data, x = column)
        plt.show()    


params = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',  'TAX',
       'PTRATIO', 'B', 'LSTAT','RAD_1' , 'RAD_2' , 'RAD_3' , 'RAD_4' , 'RAD_5' , 'RAD_6' , 'RAD_7' , 'RAD_8' , 'RAD_24']]
plt.figure(figsize=(10,10))
sns.set_theme()
sns.heatmap(params.corr(),annot=True, fmt="0.1g", cmap='PiYG')
plt.show()
'''''

inputs = data.drop('MEDV', axis=1)
targets = data['MEDV']

scalar = StandardScaler()
scalar.fit(inputs)
scaled_inputs = scalar.transform(inputs)

print(targets)