# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:25:12 2023

@author: Harini T
"""

import numpy as np
import pickle
import streamlit as st


loaded_model=pickle.load(open('D:/Projects/trained_model.sav','rb'))


#creating Function

def wine_quality_prediction(data):
    

    arr=np.asarray(data)
    data_arr=arr.reshape(1,-1)
    
    prediction=loaded_model.predict(data_arr)
    print(prediction)
    
    if(prediction[0]==1):
      return "Quality of Wine is Good"
    else:
      return "Quality of Wine is Bad"
  



def main():
    
    #Giving a title
    st.title("WineQuality Prediction Web App")
    
    #Getting Input Data From User
    
    fixed_acidity=st.text_input("Acidity level")
    
    volatile_acidity=st.text_input("Volatile Acidity level")
    
    citric_acid=st.text_input("Citric acid level")
    
    residual_sugar=st.text_input("Sugar level")
    
    chlorides=st.text_input("Chlorides level")
    
    free_sulfur_dioxide=st.text_input("Free Sulphur level")
    
    total_sulfur_dioxide=st.text_input("Total Sulphur level")
    
    density=st.text_input("Density level")
    
    pH=st.text_input("pH level")
    
    sulphates=st.text_input("Sulphates level")
    
    alcohol=st.text_input("Alcohol level")
    
   # quality=st.text_input("Quality level")
    
    
    
    #code for prediction
    diagnosis = ''
    
    #Creating a button for prediction
    
    if st.button("Check Quality"):
        diagnosis = wine_quality_prediction([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])
        
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()