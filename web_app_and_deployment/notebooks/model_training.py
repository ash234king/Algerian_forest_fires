import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('data/Algerian_forest_fires_cleaned_dataset.csv')
print(df.head())
print(df.columns)

##drop month,day and year
df.drop(['day','month','year'],axis=1,inplace=True)
print(df.head())
print(df['Classes'].value_counts())

##encoding
df['Classes']=np.where(df['Classes'].str.contains('not fire'),0,1)
print(df.head())
print(df['Classes'].value_counts())

X=df.drop('FWI',axis=1)
Y=df['FWI']
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
print(X_train.shape,X_test.shape)

## feature selection based on correlation
print(X_train.corr())

##check for multicollinearity
plt.figure(figsize=(12,10))
corr=X_train.corr()
sns.heatmap(corr,annot=True)
plt.show()

def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if(abs(corr_matrix.iloc[i,j])>threshold):
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


##set the threshold. mostly set up by domain expertise
corr_features=correlation(X_train,0.85)


##drop feature when corrleation is more than 0.85
X_train.drop(corr_features,axis=1,inplace=True)
X_test.drop(corr_features,axis=1,inplace=True)
print(X_train.shape,X_test.shape)

##feature scaling or standardization

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled)

##Box plots to understand the effect of standard scaler
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title('X_train before scaling')
plt.subplot(1,2,2)
sns.boxplot(data=X_train_scaled)
plt.title('X_train After scaling')
plt.show()

print("---------------------------------------------------------------------------")

## Linear Regression Model
print("Linear Regression")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
linreg=LinearRegression()
linreg.fit(X_train_scaled,Y_train)
y_pred=linreg.predict(X_test_scaled)
mae=mean_absolute_error(Y_test,y_pred)
score=r2_score(Y_test,y_pred)
print("my mean absolute error ",mae)
print("R2 Score ",score)
plt.scatter(Y_test,y_pred)
plt.show()

print("---------------------------------------------------------------------------")

##lasso regression
print("lasso regression")
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
lasso=Lasso()
lasso.fit(X_train_scaled,Y_train)
y_pred=lasso.predict(X_test_scaled)
mae=mean_absolute_error(Y_test,y_pred)
score=r2_score(Y_test,y_pred)
print("my mean absolute error ",mae)
print("R2 Score ",score)
plt.scatter(Y_test,y_pred)
plt.show()

print("---------------------------------------------------------------------------")

##cross validation of lasso
print("Lasso CV")
from sklearn.linear_model import LassoCV
lassocv=LassoCV(cv=5)
lassocv.fit(X_train_scaled,Y_train)
print(lassocv.alpha_)
print(lassocv.alphas_)
print(lassocv.mse_path_)
y_pred=lassocv.predict(X_test_scaled)
plt.scatter(Y_test,y_pred)
plt.show()

print("---------------------------------------------------------------------------")

##ridge regression model
print("ridge regression")
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
ridge=Ridge()
ridge.fit(X_train_scaled,Y_train)
y_pred=ridge.predict(X_test_scaled)
mae=mean_absolute_error(Y_test,y_pred)
score=r2_score(Y_test,y_pred)
print("my mean absolute error ",mae)
print("R2 Score ",score)
plt.scatter(Y_test,y_pred)
plt.show()

print("---------------------------------------------------------------------------")

##ridge cv
print("Ridge Cv")
from sklearn.linear_model import RidgeCV
ridgecv=RidgeCV(cv=5)
ridgecv.fit(X_train_scaled,Y_train)
y_pred=ridgecv.predict(X_test_scaled)
mae=mean_absolute_error(Y_test,y_pred)
score=r2_score(Y_test,y_pred)
print("my mean absolute error ",mae)
print("R2 Score ",score)
plt.scatter(Y_test,y_pred)
plt.show()


print("---------------------------------------------------------------------------")

##elastic net regression
print("elasticnet regression")
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
elastic=ElasticNet()
elastic.fit(X_train_scaled,Y_train)
y_pred=elastic.predict(X_test_scaled)
mae=mean_absolute_error(Y_test,y_pred)
score=r2_score(Y_test,y_pred)
print("my mean absolute error ",mae)
print("R2 Score ",score)
plt.scatter(Y_test,y_pred)
plt.show()

print("---------------------------------------------------------------------------")

##elasticnet cv
print("ElasticNet Cv")
from sklearn.linear_model import ElasticNetCV
elasticcv=ElasticNetCV(cv=5)
elasticcv.fit(X_train_scaled,Y_train)
y_pred=elasticcv.predict(X_test_scaled)
mae=mean_absolute_error(Y_test,y_pred)
score=r2_score(Y_test,y_pred)
print("my mean absolute error ",mae)
print("R2 Score ",score)
plt.scatter(Y_test,y_pred)
plt.show()


##pickle the machine learning models and preprocessing model i.e. standard scaler

import pickle
pickle.dump(scaler,open('C:\\Users\\yashv\\OneDrive\\Documents\\machine_learning\\ridge_and_lasso_prac\\scaler.pkl','wb'))


pickle.dump(ridge,open('C:\\Users\\yashv\\OneDrive\\Documents\\machine_learning\\ridge_and_lasso_prac\\ridge.pkl','wb'))
