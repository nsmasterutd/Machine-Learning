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

class linear_regression:
    
    def fit(self,X,y,learning_rate=0.01,iterations=1000,tolerance = 1e-3):
        
        tol = pd.DataFrame(data=np.full((X.shape[0],1),1),columns=['tol'])   
        X_new = pd.concat([X.reset_index(drop=True),tol],axis=1)
        
        no_of_data = len(y)
        theta= np.random.randn(X_new.shape[1],1) #initializes random theta 
               
        for i in range(iterations):
            
            prediction = np.dot(X_new,theta)
            
            delta = pd.DataFrame((1/no_of_data)*learning_rate*(np.dot(X_new.T,(prediction - y))))
            theta = theta-delta.iloc[:,[0]]
        
        
            #sets the 'tol' coloumn value to 1 when difference less than tolerance else 0
            delta.loc[abs(delta.iloc[:,0]) <= tolerance, 't'] = 1 
            delta.loc[abs(delta.iloc[:,0]) > tolerance, 't'] = 0 
            #if every weight value reach the minimum point based on tolerance exit the loop
        
            if sum(delta['t'])==no_of_data :
                break
        
        self.w = theta
    
    
    def predict(self,X):
        tol = pd.DataFrame(data=np.full((X.shape[0],1),1),columns=['tol']) 
       
        X_new = pd.concat([X.reset_index(drop=True),tol],axis=1)
        
        return pd.DataFrame(np.dot(X_new,self.w))



model = linear_regression()
model.fit(X_train, Y_train,iterations=500,learning_rate=0.001)

Y_train_pred = model.predict(X_train)
print("Train MSE",np.sqrt(mean_squared_error(Y_train, Y_train_pred)))
print("Train R2 Score",r2_score(Y_train,Y_train_pred))
Y_test_pred = model.predict(X_test)
print("Test MSE",np.sqrt(mean_squared_error(Y_test,Y_test_pred)))
print("Test R2 score",r2_score(Y_test,Y_test_pred))




