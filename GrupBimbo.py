# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 08:32:53 2016

@author: vemurI
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb


train = pd.read_csv('C:/Users/vemurI/Desktop/Grupo-Bimbo/train.csv',header=0,dtype  = {'Semana': 'int8',
                                                   'Agencia_ID' : 'int16',
                                                   'Canal_ID' : 'int8',
                                                   'Ruta_SAK' : 'int16',
                                                   'Cliente_ID' : 'int32',
                                                   'Producto_ID': 'int32',
                                                   'Venta_uni_hoy': 'int32',
                                                   'Venta_hoy': 'float16',
                                                   'Dev_uni_proxima':'int32',
                                                   'Dev_proxima':'float16',
                                                   'Demanda_uni_equil':'int32'}) 

test = pd.read_csv('C:/Users/vemurI/Desktop/Grupo-Bimbo/test.csv',header = 0, dtype  = {'id':'int16',
                                                    'Semana': 'int8',
                                                   'Agencia_ID' : 'int16',
                                                   'Canal_ID' : 'int8',
                                                   'Ruta_SAK' : 'int16',
                                                   'Cliente_ID' : 'int32',
                                                   'Producto_ID': 'int16'})

# Create log demand column
train['Demanda_uni_equil'] =  np.log1p(train['Demanda_uni_equil'])
data = train[['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID','Venta_uni_hoy','Dev_uni_proxima','Demanda_uni_equil']]
del train

#Inspired from Kaggler.

# Average demand of the products at each client. Can get clue of overall size of client which can be decisive?
data_pc = data.groupby(['Cliente_ID','Producto_ID'],as_index = False)['Demanda_uni_equil'].mean()
data_pc.rename(columns={'Demanda_uni_equil': 'mean_pc'}, inplace=True)

#Average demand of every Route
data = data.merge(data_pc,how='left',left_on=['Cliente_ID','Producto_ID'],
           right_on=['Cliente_ID','Producto_ID'],
           sort=True,copy=False)

ids = test['id']
test = test.drop(['id'],axis = 1)

#Split the data into train set and training evaluation set
selcols = ['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'mean_pc' ]
train_x = data[data['Semana'] <= 8][selcols]
target_y = data[data['Semana'] <= 8]['Demanda_uni_equil']

# Week 9 is training error evaluation set
eval_train_x = data[data['Semana'] == 9][selcols]
eval_target_y = data[data['Semana'] == 9 ]['Demanda_uni_equil']

xlf = xgb.XGBRegressor(max_depth=4, 
                        learning_rate=0.20, 
                        n_estimators=20, 
                        silent=True, 
                        objective='reg:linear', 
                        nthread=3, 
                        seed=7,
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        missing=None)

xlf.fit(train_x,target_y,eval_metric='rmse',eval_set=[(eval_train_x,eval_target_y)],early_stopping_rounds=50)

preds = xlf.predict(test)
pred = np.exp(preds)-1

submission = pd.DataFrame({ 'Demanda_uni_equil': pred,'id':ids})
test.to_csv('C:/Users/vemurI/Desktop/Grupo-Bimbo/Pred.csv', index=False)