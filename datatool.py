import pandas as pd
import sys
import streamlit as st
from streamlit.config import _global_development_mode
from sklearn.preprocessing import LabelEncoder,StandardScaler

global list_globaldf


@st.cache
def fnc_globaldf():
    global globaldf
 

def displaydataframe(df):
    st.dataframe(df)
    
# @st.cache
def drop_col(col):
    try:
        fnc_globaldf.globaldf=fnc_globaldf.globaldf.drop(columns=col)
        return fnc_globaldf.globaldf
    except:
        e=sys.exc_info()
        return st.error(e)

def dropna_col(axis=0,how='any', thresh=None, subset=None):
    try:
        fnc_globaldf.globaldf=fnc_globaldf.globaldf.dropna(axis,how, thresh, subset)
        return fnc_globaldf.globaldf
    except:
        e=sys.exc_info()
        return st.error(e)
    


def fillna_col(col):
    try:
        fnc_globaldf.globaldf[col]=fnc_globaldf.globaldf[col].fillna(fnc_globaldf.globaldf[col].mean())
        return fnc_globaldf.globaldf
    except:
        e=sys.exc_info()
        return st.error(e)
    
def labelencoder(col):
    try:
        label_encoder = LabelEncoder()
        for i in col: 
            fnc_globaldf.globaldf[i]= label_encoder.fit_transform(fnc_globaldf.globaldf[i])
        return fnc_globaldf.globaldf
    except:
        e=sys.exc_info()
        return st.error(e) 
        
def standardscaler(col):
    try:
        scaler=StandardScaler()
        scaler.fit(fnc_globaldf.globaldf[col])
        scaler_features=pd.DataFrame(scaler.transform(fnc_globaldf.globaldf[col]),columns=col)
        non_scaler_col=(fnc_globaldf.globaldf.columns).difference(col)
        fnc_globaldf.globaldf=pd.concat([scaler_features,fnc_globaldf.globaldf[non_scaler_col]],axis=1)
        return fnc_globaldf.globaldf
    except:
        e=sys.exc_info()
        return st.error(e)