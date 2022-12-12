import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import pickle

df_Encoded = pd.read_csv('Flight_Trained.csv')
df_Encoded.drop('Unnamed: 0',axis=1,inplace=True)

X = df_Encoded.drop('Price',axis=1)
y = df_Encoded['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 142)
ET = ExtraTreesRegressor(n_estimators=250)
ET_model = ET.fit(X_train, y_train)



pickle.dump(ET_model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

