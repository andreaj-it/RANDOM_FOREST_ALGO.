import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf
import seaborn as sns 

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#imports para el ejecricio nuevo de forest 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier 
 
from sklearn import metrics

#Step 1

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv')

#Step 2

df=df.drop(columns=['Name'])

df['Sex'] = df['Sex'].map({'male':1,'female':0})

df=df.drop(columns=['Ticket'])

df=df.drop(columns=['Cabin'])

df['Embarked'] = df['Embarked'].map({'S':2,'C':1,'Q':0})

df['Age'][np.isnan(df['Age'])]=df['Age'].mean()

df['Embarked'][np.isnan(df['Embarked'])]=2

#Step 3

x = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=53, test_size=0.2)


modelo = RandomForestClassifier(n_estimators=50, random_state=53)
modelo.fit(X_train, y_train)
y_train_pred = modelo.predict(X_train)
y_test_pred = modelo.predict(X_test)

train_report = metrics.classification_report(y_pred=y_train_pred, y_true=y_train)
print(train_report)

train_report = metrics.classification_report(y_pred=y_test_pred, y_true=y_test)
print(train_report)


#voy de nuevo todo, con otro valor !!! 

modelo2 = RandomForestClassifier(n_estimators=5,random_state=53, max_leaf_nodes=20, max_depth=10)

modelo2.fit(X_train, y_train)

train_report = metrics.classification_report(y_pred=y_train_pred, y_true=y_train)
print(train_report)

test_report = metrics.classification_report(y_pred=y_test_pred, y_true=y_test)
print(test_report)

#Step 5

import pickle

filename = '../models/finalized_model.sav'
pickle.dump(modelo2, open(filename, 'wb'))
