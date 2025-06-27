import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("data/Algerian_forest_fires_dataset_UPDATE.csv",header=1)
print(df.head())
print(df.info())

### DATA Cleaning
print(df[df.isnull().any(axis=1)])

df.loc[:122,"Region"]=0
df.loc[122:,"Region"]=1
print(df.info())
df[['Region']]=df[['Region']].astype(int)

## Remove the null values
df=df.dropna().reset_index(drop=True)
print(df.isnull().sum())

print(df.iloc[[122]])
df=df.drop(122).reset_index(drop=True)
print(df.iloc[[122]])

print(df.columns)
### fix spaces in column names

df.columns=df.columns.str.strip()
print(df.columns)

### change the required columns as integer datatype

df[['month','day','year','Temperature','RH','Ws']]=df[['month','day','year','Temperature','RH','Ws']].astype(int)
print(df.info())

## changing the other columns to float datatype

objects=[features for features in df.columns if df[features].dtypes=='O']
for i in objects:
    if i!='Classes':
        df[i]=df[i].astype(float)
print(df.info())
print(df.describe())

## lets save the clean dataset
df.to_csv('data/Algerian_forest_fires_cleaned_dataset.csv',index=False)
df_copy=df.drop(['day','month','year'],axis=1)
print(df_copy.head())
print(df_copy['Classes'].value_counts())

## encoding of the categories in the classes
df_copy['Classes']=np.where(df_copy['Classes'].str.contains('not fire'),0,1)
print(df_copy.head())
print(df_copy['Classes'].value_counts())

plt.style.use('seaborn-v0_8')  # or 'seaborn-v0_8-whitegrid' etc.
df_copy.hist(bins=50, figsize=(20, 15))
plt.show()

## percentage for pie chart
percentage=df_copy['Classes'].value_counts(normalize=True)*100

## plotting piechart
classlabels=["Fire","Not Fire"]
plt.figure(figsize=(12,7))
plt.pie(percentage,labels=classlabels,autopct='%1.1f%%')
plt.title("Pie chart of classes")
plt.show()

##correlation
print(df_copy.corr())
sns.heatmap(df_copy.corr())
plt.show()

##box plots
sns.boxplot(df_copy['FWI'],color='g')
plt.show()

df['Classes']=np.where(df['Classes'].str.contains('not fire'),0,1)

##monthly fire analysis
dftemp=df.loc[df['Region']==1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data=df)
plt.ylabel('Number of fires',weight='bold')
plt.xlabel('Months',weight='bold')
plt.title("Fire Analysis of Sidi-Bel Regions",weight='bold')
plt.show()

dftemp=df.loc[df['Region']==0]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data=df)
plt.ylabel('Number of fires',weight='bold')
plt.xlabel('Months',weight='bold')
plt.title("Fire Analysis of Bejia Regions",weight='bold')
plt.show()