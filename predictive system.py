# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import numpy as np

loaded_model=pickle.load(open('D:/Projects/trained_model.sav','rb'))


data=[7.5,2.5,3.09,3.81,6.2,8.3,0.9978,0.2,2.7,10.5,2.5]
arr=np.asarray(data)
data_arr=arr.reshape(1,-1)
prediction=loaded_model.predict(data_arr)
print(prediction)
if(prediction[0]==1):
  print("Quality of Wine is Good")
else:
  print("Quality of Wine is Bad")