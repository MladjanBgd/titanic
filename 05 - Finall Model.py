# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:44:39 2022

@author: mladjan.jovanovic
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_ds = pd.read_csv('./train.csv')
test_ds = pd.read_csv('./test.csv')


print(train_ds.head(15))

print(train_ds.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  //drop, random
 1   Survived     891 non-null    int64  -*-
 2   Pclass       891 non-null    int64  //keep?
 3   Name         891 non-null    object //make Title and drop
 4   Sex          891 non-null    object //switch to 0,1
 5   Age          714 non-null    float64 //impute with median for Pclass
 6   SibSp        891 non-null    int64  //make FamSize and drop
 7   Parch        891 non-null    int64  //make FamSize and drop
 8   Ticket       891 non-null    object //drop, random
 9   Fare         891 non-null    float64 //map log f
 10  Cabin        204 non-null    object  //drop, full of NaN's'
 11  Embarked     889 non-null    object //make dummys S,Q,T
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None
"""
plt.subplots(figsize=(9,5))
sns.heatmap(train_ds.isna(), yticklabels=False, cbar=False, cmap='YlOrRd' )


train_ds = train_ds.drop(columns='Cabin') #full of NaN's

train_ds=train_ds.drop(columns='PassengerId') #I dont think it is important

print(train_ds['Ticket'].nunique())
train_ds=train_ds.drop(columns='Ticket') #seems randomly

train_ds['Embarked']=train_ds['Embarked'].replace(np.nan,train_ds['Embarked'].mode()[0]) #fill NaN with most frequent

sns.catplot(data=train_ds, x='Pclass', col='Sex', kind='count')

sns.catplot(data=train_ds, x='Embarked', col='Survived', kind='count')

plt.subplots(figsize=(9,5))
sns.heatmap(data=train_ds[['Survived','Pclass','Age','SibSp','Parch','Fare']].corr(), cmap='Blues')

train_ds['Sex']=train_ds['Sex'].replace({'male':0,'female':1}) #map male/female 1/0

#fix 11. Embarkeg
embark=pd.get_dummies(train_ds['Embarked'])
train_ds = pd.concat([train_ds,embark], axis=1)
train_ds = train_ds.drop(columns='Embarked')

#fix 9. Fare
plt.subplots(figsize=(20,10))
sns.histplot(data=train_ds['Fare'], bins=60, color='g')
train_ds['Fare']=train_ds['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
plt.subplots(figsize=(20,10))
sns.histplot(data=train_ds['Fare'], bins=60, color='g')

#fix 6. SibSp and 7. Parch
train_ds['FamSize']= train_ds['SibSp'] + train_ds['Parch'] + 1
train_ds = train_ds.drop(columns=['SibSp','Parch'])

#fix 3. Name
title=[i.split(', ')[1].split('.')[0] for i in train_ds['Name']]
print(list(dict.fromkeys(title)))
"""
['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess', 'Jonkheer']
"""
train_ds['Title'] = pd.Series(title)
train_ds["Title"] = train_ds["Title"].replace(['Lady','the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
train_ds["Title"] = train_ds["Title"].replace('Mlle','Miss')
train_ds["Title"] = train_ds["Title"].replace('Ms','Miss')
train_ds["Title"] = train_ds["Title"].replace('Mme','Mrs')

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
end_df = pd.DataFrame(enc.fit_transform(train_ds[['Title']]).toarray())
train_ds = train_ds.join(end_df)
train_ds = train_ds.drop(columns='Title')

train_ds = train_ds.drop(columns='Name')


#fix 5. Age

print(train_ds.groupby(['Pclass']).mean())
"""
        Survived       Sex        Age  ...         Q         S   FamSize
Pclass                                 ...                              
1       0.629630  0.435185  38.233441  ...  0.009259  0.597222  1.773148
2       0.472826  0.413043  29.877630  ...  0.016304  0.891304  1.782609
3       0.242363  0.293279  25.140620  ...  0.146640  0.718941  2.008147
"""

# train_ds['Age']=train_ds['Age'].map(lambda i: 38 if ('Pclass' == 1) else i)
# train_ds['Age']=train_ds['Age'].map(lambda i: 30 if ('Pclass' == 2) else i)
# train_ds['Age']=train_ds['Age'].map(lambda i: 25 if ('Pclass' == 3) else i)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age

train_ds['Age'] = train_ds[['Age','Pclass']].apply(impute_age, axis=1)

sns.heatmap(data=train_ds.isna())

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1,2,3,5], "min_samples_split" : [10,11,12,13], "n_estimators": [350, 400, 450, 500,550], "max_depth":[6,7,8,9]}

gs=GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

X = train_ds.iloc[:, 1:]
y = train_ds.iloc[:, 0]
gs=gs.fit(X, y)
print(gs)



























