import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#Gradient boosting regression (GBR)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle

df = pd.read_csv('dataset.csv')
y = np.asarray(df['Eg'])
z = np.asarray(df['Ehull'])
a = []
for x in range(len(y)):
  if y[x] == 0 or z[x]==0:
    a.append(x)
print(a)
a.reverse()
df.drop(a, axis=0, inplace=True)
df.shape
x=df[['Ef', 'rho', 'of', 'gtf', 'ea_mean', 'spec_heat_mean', 'heat_fus_mean',
      'vdw_mean', 'atom_rad_mean', 'av_ionrad_mean', 'av_rsp_mean',
      'x_mean']]
df['Eg'] = df['Eg'].astype('float64') 
#y = np.asarray(df['Eg'])
y = df[['Ehull', 'Eg']]
X_train1, X_test1, y_train1, y_test1 = train_test_split( x, y, test_size=0.2, random_state=25)
gbr = GradientBoostingRegressor()
modelMOR = MultiOutputRegressor(estimator=gbr)
modelMOR.fit(X_train1, y_train1)

with open('model.pickle','wb') as f:
    pickle.dump(modelMOR,f)
model=pickle.load(open('model.pickle','rb'))
final = [30.0, 25.0, 30.0, 25.0, 20.0, 30.0, 30.0, 1.0, 1.0, 1.0, 25.0, 30.0]
ehull, eg = model.predict([final])[0]
print(ehull)
print(eg)
