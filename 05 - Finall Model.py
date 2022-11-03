# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:44:39 2022

@author: mladjan.jovanovic
"""
#[1] make pipeline for transformation train_ds and test_ds

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_ds = pd.read_csv('./train.csv')
pred_ds = pd.read_csv('./test.csv')


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

#fix fetaure names type
train_ds.columns=train_ds.columns.astype(str)

##########-make gridsearchcv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

X = train_ds.iloc[:, 1:]
y = train_ds.iloc[:, 0]

# #########-
# rf=RandomForestClassifier()

# param_grid = {"n_estimators": [400, 450, 480, 490, 500, 600, 700, 800, 1000],
#               "criterion" : ["gini", "entropy"],
#               "max_depth": [12,13,14,15],
#               "min_samples_split": [16,17,18,19,20],
#               "min_samples_leaf": [1,2,3,5]}

# gs=GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
# gs=gs.fit(X, y)

# print(gs.best_params_)

# """
# {'criterion': 'entropy', 'max_depth': 12, 'min_samples_leaf': 2, 'min_samples_split': 19, 'n_estimators': 500}
# """
# #########-end of make gridsearchcv


##########-make finall model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

rf=RandomForestClassifier(criterion='entropy', max_depth=12, min_samples_leaf=2, min_samples_split=19, n_estimators=500)
#fit model
rf.fit(X_train, y_train)

y_t_pred=rf.predict(X_test)

#what are scores
from sklearn.metrics import classification_report

clr = classification_report(y_test, y_t_pred)
print(clr)
##########-make finall model

#quick fix of pred_ds, make pipline in next refactoring [1]
PID=pred_ds.pop('PassengerId')
pred_ds = pred_ds.drop(columns='Cabin') #full of NaN's
pred_ds=pred_ds.drop(columns='Ticket') #seems randomly
pred_ds['Embarked']=pred_ds['Embarked'].replace(np.nan,pred_ds['Embarked'].mode()[0]) #fill NaN with most frequent
pred_ds['Sex']=pred_ds['Sex'].replace({'male':0,'female':1}) #map male/female 1/0
embark=pd.get_dummies(pred_ds['Embarked'])
pred_ds = pd.concat([pred_ds,embark], axis=1)
pred_ds = pred_ds.drop(columns='Embarked')
pred_ds['Fare']=pred_ds['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
pred_ds['FamSize']= pred_ds['SibSp'] + pred_ds['Parch'] + 1
pred_ds = pred_ds.drop(columns=['SibSp','Parch'])
title=[i.split(', ')[1].split('.')[0] for i in pred_ds['Name']]
pred_ds['Title'] = pd.Series(title)
pred_ds["Title"] = pred_ds["Title"].replace(['Lady','the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
pred_ds["Title"] = pred_ds["Title"].replace('Mlle','Miss')
pred_ds["Title"] = pred_ds["Title"].replace('Ms','Miss')
pred_ds["Title"] = pred_ds["Title"].replace('Mme','Mrs')
end_df = pd.DataFrame(enc.fit_transform(pred_ds[['Title']]).toarray())
pred_ds = pred_ds.join(end_df)
pred_ds = pred_ds.drop(columns='Title')
pred_ds = pred_ds.drop(columns='Name')
pred_ds['Age'] = pred_ds[['Age','Pclass']].apply(impute_age, axis=1)
pred_ds.columns=pred_ds.columns.astype(str)
X_test = pred_ds.iloc[:,:]

#make predictions
y_pred = rf.predict(X_test)

#make finall DataFrame
res=pd.DataFrame(PID)
res['Survived']=y_pred

#write to file
res_f = res.to_csv('./gender_submission.csv', index=False)
































