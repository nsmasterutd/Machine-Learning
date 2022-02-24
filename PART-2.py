import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data")
df = pd.DataFrame(df)
df.columns = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]
# print(df.info())
del df["normalized-losses"]
# print(df.info())


print(df['price'].value_counts())
price = df['price'].loc[df['price'] != '?']
mean_price = price.astype(int).mean()
df['price'] = df['price'].replace('?',mean_price).astype(int)
print(df['price'].value_counts())

print(df['horsepower'].value_counts())
horsepower = df['horsepower'].loc[df['horsepower'] != '?']
mean_horsepower = horsepower.astype(int).mean()
df['horsepower'] = df['horsepower'].replace('?',mean_horsepower).astype(int)
print(df['horsepower'].value_counts())

# Fields that can not be taken average of
print(df.info())

df['bore'] = pd.to_numeric(df['bore'],errors='coerce')

df['stroke'] = pd.to_numeric(df['stroke'],errors='coerce')

df['peak-rpm'] = pd.to_numeric(df['peak-rpm'],errors='coerce')

df = df[df['num-of-doors'] != '?']

df=df.dropna()
df=df.drop_duplicates()

print(df.info())

for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes

############################## END OF EDA #############################################

# Scaling

s=StandardScaler()
X = df.iloc[:,:24]
X = pd.DataFrame(s.fit(X).fit_transform(X))
Y = df.iloc[:,[24]]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

#Model Making

from sklearn import linear_model
from sklearn import metrics

Y_train = np.ravel(Y_train)
Y_test = np.ravel(Y_test) #Flatten the target variable so that it can be fitted into model

reg = linear_model.SGDRegressor(max_iter=500,learning_rate='constant',eta0=0.001,tol=1e-3)
reg.fit(X_train,Y_train)
# Used Stochastic Gradient Descent so that we can fit different values of learning rate and interations

Y_pred = reg.predict(X_train)
print("For Training:")
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_train, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_train, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_train, Y_pred)))

Y_pred2 = reg.predict(X_test)
print("For Testing: ")
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred2)))
