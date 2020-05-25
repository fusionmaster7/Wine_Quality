# %%
#NumPy
import numpy as np
#Pandas
import pandas as pd
from sklearn.model_selection import train_test_split
#Preprocessing modules
from sklearn import preprocessing
#Import Random forest module
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
import joblib

url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url,sep=';')

y = df.quality
X = df.drop('quality',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123,stratify=y)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

piperline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=100))
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(piperline,hyperparameters,cv=10)
clf.fit(X_train,y_train)
print(clf.best_params_)

y_pred = clf.predict(X_test)
print(r2_score(y_test,y_pred))

print(mean_squared_error(y_test,y_pred))

joblib.dump(clf,'rf_regressor.pkl')


# %%


