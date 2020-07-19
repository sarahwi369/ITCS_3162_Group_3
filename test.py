import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

print('import')
sns.set(style='ticks',color_codes=True)

print('set')
url = 'https://raw.githubusercontent.com/Ayushijain09/Regression-on-COVID-dataset/master/COVID-19_Daily_Testing.csv'
print('url')
data = pd.read_csv(url)
print('data')
data.head()
print('head')

print(data.info())

data['Cases'] = data['Cases'].str.replace(',', '')
data['Tests'] = data['Tests'].str.replace(',', '')
data['Cases'] = pd.to_numeric(data['Cases'])
data['Tests'] = pd.to_numeric(data['Tests'])
print('data stuff')

data_numeric = data.select_dtypes(include=['float64', 'int64'])
print('data_numeric')
plt.figure(figsize=(20,10))
print('figure')
sns.pairplot(data_numeric)
print('pairplot')
plt.show()
print('show')
