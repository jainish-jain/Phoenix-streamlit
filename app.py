from analytics import firestore
from inspect import Traceback
from pickle import NONE
import pickle
from random import random, shuffle
from altair.vegalite.v4.api import value
from altair.vegalite.v4.schema.channels import Column
from numpy.core.fromnumeric import choose
from numpy.lib.npyio import save
import pandas,sys
from scipy.sparse import data
import streamlit as st
import pandas as pd
import streamlit
import models,sklearn
import datatool,base64,os
import matplotlib
import matplotlib.pyplot as plt
from datatool import  displaydataframe,fnc_globaldf,drop_col,fillna_col,dropna_col, labelencoder, standardscaler
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from bokeh.models.widgets import Div
import numpy as np
import io
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder
import analytics
import json,datetime


#Models
from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


st.set_page_config(page_title="Phoenix" ,page_icon="favicon.png")

st.markdown(
    """
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Belleza" />
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-family: "Belleza", sans-serif;
            # font-weight:10 !important;
            font-size:40px !important;
            #color: #f9a01b !important;
            padding-top: 12px !important;
            padding-left: 12px !important;
        }
        .des-text {
            font-family: "Belleza", sans-serif;
            font-size:18px !important;
            padding-top: 12px !important;
            padding-left: 12px !important;
            padding-right: 12px !important;  
        }
        .logo-img {
            float:right;
            width:90px;
            height:80px;
        }
        </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open("favicon.png", "rb").read()).decode()}">
        <p class="logo-text"> Phoenix Workspace</p><br>
    </div>
    <p class="des-text"> A holistic approach towards data processing - generalized machine learning application. An application to process, filter, cleanse and visualize data on univariate and multivariate graphs.</p> <br>   
    """,
    unsafe_allow_html=True,
)
def aggriddf(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False)
    gb.configure_grid_options(domLayout='normal')
    #gb.configure_side_bar()
    gridOptions = gb.build()
    grid_response = AgGrid(
    df, 
    gridOptions=gridOptions,
    height=330, 
    width='100%',
    #data_return_mode=return_mode_value, 
    #update_mode=update_mode_value,
    allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
    enable_enterprise_modules=False,
    )
    return grid_response

def upload_csv():
    csv_file=None
    csv_file = st.file_uploader("Upload CSV Dataset", type=([".csv"]))
    col_names_csv=[]
    dfcsv=None
    #   fnc_globaldf.globaldf=None
    if csv_file:
        dfcsv=pd.read_csv(csv_file)
        st.write(f"**Dataframe: {csv_file.name}**")
        #st.dataframe(dfcsv)  
        csv_aggrid=aggriddf(dfcsv)
        col_names_csv=list(dfcsv.columns)
        try:
            st.write("**dataframe.describe( )**")
            st.dataframe(dfcsv.describe())
            st.write("**dataframe.info( )**")
            buffer = io.StringIO()
            dfcsv.info(buf=buffer)
            dfinfo = buffer.getvalue()
            st.text(dfinfo)
        except:pass
        fnc_globaldf.globaldf=dfcsv.copy()

        return (dfcsv,col_names_csv,csv_file.name)
    return False


def btndf(btnname,link):
    custom_css = f""" 
        <style>
            #button_id{{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #button_id:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #button_id:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a id="button_id" href={link} target="_blank">{btnname}</a>'
    return dl_link


if __name__=="__main__":

    
    tmp_csv=upload_csv()     

    data_tool_level=st.sidebar.subheader("Data Tools")
    data_tool_level1=st.sidebar.checkbox('Processing and Cleaning')
    data_tool_level2=st.sidebar.checkbox('Exploratory Data Analysis')
    
    
    st.sidebar.text("\n")
    
    st.sidebar.markdown(btndf('To Run Custom Python Code','https://share.streamlit.io/jainish-jain/phoenix/main/code.py'),unsafe_allow_html=True)
       
    st.sidebar.markdown("<br>"*5,unsafe_allow_html=True)
    st.sidebar.subheader("Connect with me :")
    st.sidebar.markdown(btndf('Github','https://github.com/Jainish-Jain'),unsafe_allow_html=True)
    #st.sidebar.markdown("[![GitHub followers](https://img.shields.io/github/followers/Jainish-Jain?label=%40Jainish-Jain&style=social)](https://github.com/Jainish-Jain)",unsafe_allow_html=True)
    st.sidebar.text("")
    st.sidebar.markdown(btndf('LinkedIn','https://linkedin.com/in/jainish-jain'),unsafe_allow_html=True)


    if data_tool_level1 and tmp_csv!=False:
        #st.subheader("Data Tool")
        check1=st.beta_expander("Data Processing and  Cleaning Tools:",expanded=True)
        if check1.checkbox("drop"):
            col=check1.multiselect("Choose Column",[*fnc_globaldf.globaldf.columns],key="data_drop")
            fnc_globaldf.globaldf=drop_col(col)
        if check1.checkbox("fillna: mean"):
            col=check1.multiselect("Choose Column",[*fnc_globaldf.globaldf.columns],key="fillna")
            fnc_globaldf.globaldf=fillna_col(col)
        if check1.checkbox("dropna"):
            axis_ip=check1.selectbox("axis",[0,1])
            thresh_ip=check1.text_input("thresh","None")
            col=check1.multiselect("Choose Column",[*fnc_globaldf.globaldf.columns],key="dropna")
            how_ip=check1.selectbox("how",["any","all"])
            fnc_globaldf.globaldf=dropna_col(axis=axis_ip,thresh=float(thresh_ip) if not thresh_ip else None ,how=how_ip,subset=col)
        if check1.checkbox("notna"):
            check1.text("Entries in not Na")
            check1.dataframe(fnc_globaldf.globaldf.notna())
        if check1.checkbox("unique"):
            col=check1.selectbox("Choose Column",[*fnc_globaldf.globaldf.columns],key="unique")
            check1.dataframe(fnc_globaldf.globaldf[col].unique())
        if check1.checkbox("labelencoder"):
            col=check1.multiselect("Choose Column",[*fnc_globaldf.globaldf.columns],key="labelencoder")
            fnc_globaldf.globaldf=labelencoder(col)
        if check1.checkbox("standardscaler"):
            col=check1.multiselect("Choose Column",[*fnc_globaldf.globaldf.columns],key="scaler")
            if len(col)!=0:
                fnc_globaldf.globaldf=(standardscaler(col)) 

        
        
        
        try:
            check1.write("**Dataframe :**")
            check1.dataframe(fnc_globaldf.globaldf)           
            check1.write("**dataframe.describe( )**")
            check1.dataframe(fnc_globaldf.globaldf.describe())
            check1.write("**dataframe.info( )**")
            buffer = io.StringIO()
            fnc_globaldf.globaldf.info(buf=buffer)
            dfinfo = buffer.getvalue()
            check1.text(dfinfo)

            csv = fnc_globaldf.globaldf.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            custom_css = f""" 
            <style>
                #btncsv {{
                    background-color: rgb(255, 255, 255);
                    color: rgb(38, 39, 48);
                    padding: 0.25em 0.38em;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: rgb(230, 234, 241);
                    border-image: initial;
                }} 
                #btncsv:hover {{
                    border-color: rgb(246, 51, 102);
                    color: rgb(246, 51, 102);
                }}
                #btncsv:active {{
                    box-shadow: none;
                    background-color: rgb(246, 51, 102);
                    color: white;
                    }}
            </style> """
            href = custom_css+ f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ".csv")'
            dl_link = custom_css + f'<a download="{(tmp_csv[2])[:-4]}.csv" id="btncsv" href="data:file/txt;base64,{b64}">Download CSV</a><br></br>'
            check1.markdown(dl_link, unsafe_allow_html=True)
        except:pass  
        

        
        


    if data_tool_level2 and tmp_csv!=False:
        check2=st.beta_expander("Exploratory Data Analysis :",expanded=True)
        if check2.checkbox('line_chart'):
            x,y,c,l=check2.beta_columns(4)
            col2x=x.selectbox("Choose X ",["",*fnc_globaldf.globaldf.columns],key="line_chartx")
            col2y=y.selectbox("Choose Y ",["",*fnc_globaldf.globaldf.columns],key="line_charty")
            col2c=c.selectbox("Choose Color",[None,*fnc_globaldf.globaldf.columns],key="line_chartc")
            col2l=l.selectbox("Choose Line Group",[None,*fnc_globaldf.globaldf.columns],key="line_chartc")
            
            try:
                fig=check2.plotly_chart(px.line(fnc_globaldf.globaldf,x=col2x,y=col2y,color=col2c,line_group=col2l)) if col2x and col2y else None
            except:
                e=sys.exc_info()
                check2.error(e)

        if check2.checkbox('pie_chart'):
            v,n=check2.beta_columns(2)
            col2v=v.selectbox("Choose Value",["",*fnc_globaldf.globaldf.columns],key="pie_chartv")
            col2n=n.selectbox("Choose Name",["",*fnc_globaldf.globaldf.columns],key="pie_chartn")
            
            try:
                fig=check2.plotly_chart(px.pie(fnc_globaldf.globaldf,values=col2v,names=col2n)) if col2v and col2n else None
                
            except:
                e=sys.exc_info()
                check2.error(e)

        if check2.checkbox('scatter_plot'):
            x,y,c,s=check2.beta_columns(4)
            col2x=x.selectbox("Choose X ",["",*fnc_globaldf.globaldf.columns],key="scatter_plotx")
            col2y=y.selectbox("Choose Y ",["",*fnc_globaldf.globaldf.columns],key="scatter_ploty")
            col2c=c.selectbox("Choose Color",[None,*fnc_globaldf.globaldf.columns],key="scatter_plotc")
            col2s=s.selectbox("Choose Size",[None,*fnc_globaldf.globaldf.columns],key="scatter_plots")
            
            try:
                fig=check2.plotly_chart(px.scatter(fnc_globaldf.globaldf,x=col2x,y=col2y,color=col2c,size=col2s)) if col2x and col2y else None
            except:
                e=sys.exc_info()
                check2.error(e)


        if check2.checkbox('scatter_matrix'):
            col2=check2.multiselect("Choose",["",*fnc_globaldf.globaldf.columns],key="scatter_matrix") 
            try:
                fig=check2.plotly_chart(px.scatter_matrix(fnc_globaldf.globaldf[col2]))
            except:
                e=sys.exc_info()
                check2.error(e)

        # if check2.checkbox('area_chart'):
        #     col2=check2.multiselect("Choose Column",[*fnc_globaldf.globaldf.columns],key="area_chart")
        #     check2.area_chart(fnc_globaldf.globaldf[col2])

        if check2.checkbox('bar_chart'):
            x,y,c,b=check2.beta_columns(4)
            col2x=x.selectbox("Choose X ",["",*fnc_globaldf.globaldf.columns],key="bar_chartx")
            col2y=y.multiselect("Choose Y ",["",*fnc_globaldf.globaldf.columns],key="bar_charty")
            col2c=c.selectbox("Choose Color",[None,*fnc_globaldf.globaldf.columns],key="bar_chartc")
            col2b=b.selectbox("Choose Barmode",["stack", "group", "overlay", "relative"])
            if len(col2y)!=0:
                try:
                    fig=check2.plotly_chart(px.bar(fnc_globaldf.globaldf,x=col2x,y=col2y,color=col2c,barmode=col2b)) if col2x and col2y else None
                except:
                    e=sys.exc_info()
                    check2.error(e)

        if check2.checkbox('histogram'):
            x,y,c,m=check2.beta_columns(4)
            col2x=x.selectbox("Choose X ",["",*fnc_globaldf.globaldf.columns],key="histogramx")
            col2y=y.selectbox("Choose Y ",["",*fnc_globaldf.globaldf.columns],key="histogramy")
            col2c=c.selectbox("Choose Color",[None,*fnc_globaldf.globaldf.columns],key="histogramc")
            col2m=m.selectbox("Choose Marginal",[None,"rug", "box", "violin"])
            try:
                fig=check2.plotly_chart(px.histogram(fnc_globaldf.globaldf,x=col2x,y=col2y,color=col2c,marginal=col2m)) if col2x and col2y else None
            except:
                e=sys.exc_info()
                check2.error(e)


        if check2.checkbox('box_plot'):
            x,y,c,p=check2.beta_columns(4)
            col2x=x.selectbox("Choose X ",["",*fnc_globaldf.globaldf.columns],key="box_plotx")
            col2y=y.selectbox("Choose Y ",["",*fnc_globaldf.globaldf.columns],key="box_ploty")
            col2c=c.selectbox("Choose Color",[None,*fnc_globaldf.globaldf.columns],key="box_plotc")
            col2p=p.selectbox("Choose Marginal",[None,'all', 'outliers', 'suspectedoutliers'])
            try:
                fig=check2.plotly_chart(px.box(fnc_globaldf.globaldf,x=col2x,y=col2y,color=col2c,points=col2p)) if col2x and col2y else None
            except:
                e=sys.exc_info()
                check2.error(e)

        if check2.checkbox('heatmap'):
            col2=check2.multiselect("Choose",["",*fnc_globaldf.globaldf.columns],key="heatmap")
            try:
                fig=check2.plotly_chart(px.imshow(fnc_globaldf.globaldf[col2]))
            except:
                e=sys.exc_info()
                check2.error(e)

        # if check2.checkbox('plotly_chart'):
        #     col2=check2.multiselect("Choose Column",[*fnc_globaldf.globaldf.columns],key="plotly_chart")
        #     check2.plotly_chart(fnc_globaldf.globaldf[col2])
        # if check2.checkbox('bokeh_chart'):
        #     col2=check2.multiselect("Choose Column",[*fnc_globaldf.globaldf.columns],key="bokeh_chart")
        #     check2.bokeh_chart(fnc_globaldf.globaldf[col2])

        # if check2.checkbox('graphviz_chart'):
        #     col2=check2.multiselect("Choose Column",[*fnc_globaldf.globaldf.columns],key="graphivz_chart")
        #     check2.graphviz_chart(fnc_globaldf.globaldf[col2])
    







    m=[[]]
    supervised_models_list={1:'Linear Regression',2:'Logistic Regression',3:'Naïve Bayes: GaussianNB',4:'K-Nearest Neighbours',5:'Decision Tree Classifier',6:'Support Vector Machine Classifier'}
    un_supervised_models_list={7:"K Means Clustering",8:"Principal Component Analysis"}
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("<h3>Choose Algorithm</h3>",unsafe_allow_html=True)
    choose_algo=st.selectbox("",['',*supervised_models_list.values(),*un_supervised_models_list.values()])
    model_detail=st.beta_expander("Documentation")
    if choose_algo == supervised_models_list[1]:
        model_detail.help(sklearn.linear_model.LinearRegression)
    elif choose_algo == supervised_models_list[2]:
        model_detail.help(sklearn.linear_model.LogisticRegression)
    elif choose_algo==supervised_models_list[3]:
        model_detail.help(sklearn.naive_bayes.GaussianNB)
    elif choose_algo==supervised_models_list[4]:
        model_detail.help(sklearn.neighbors.KNeighborsClassifier)
    elif choose_algo==supervised_models_list[5]:
        model_detail.help(sklearn.tree.DecisionTreeClassifier)
    elif choose_algo==supervised_models_list[6]:
        model_detail.help(sklearn.svm.SVC)
    elif choose_algo==un_supervised_models_list[7]:
        model_detail.help(sklearn.cluster.KMeans)
    elif choose_algo==un_supervised_models_list[8]:
        model_detail.help(sklearn.decomposition.PCA)
    else:pass

    try:
        #fnc_globaldf.globaldf
        if choose_algo and tmp_csv==False:
            st.warning("File not found, Please upload a file")
        elif choose_algo and tmp_csv!=False and len(list(fnc_globaldf.globaldf.columns))!=0:
            df=fnc_globaldf.globaldf.copy()
            col_names=list(df.columns)

            
            '''\n\n'''
            pkl_model=False
            if st.checkbox("Use Pickle Model"):
                # def pkl_file_upload():
                #     pkl_file=st.file_uploader("Upload Pickle Model", type=([".pkl"]),key='pkl')
                #     return pkl_file
                # st.subheader("Test Pickle Model")
                pkl_file=st.file_uploader("Upload Pickle Model", type=([".pkl"]),key='pkl')#pkl_file_upload()
                pkl_model=(pickle.load(pkl_file)) if pkl_file is not None else False

            dep_col,tar_col=st.beta_columns(2)
            dep=dep_col.multiselect('Dependent Variables',col_names)
            tar=tar_col.selectbox('Target Variable',[" ",*col_names]) if choose_algo not in un_supervised_models_list.values() else None
            #validation_val=X_train=X_test=Y_train=Y_test=None
            

            # check3=st.beta_expander("Model Validation :",expanded=True)       
            
            dep_shape=dep_col.write(df[dep].shape) if dep else None
            tar_shape=tar_col.write(df[tar].shape) if (tar and tar!=" ") else None
            
            X_train=df[dep]
            Y_train=df[tar] if (tar and tar!=" ") else False
            X_test=Y_test=tts=False
            
            dv=st.markdown("<h3>Data Validation: </h3>",unsafe_allow_html=True) if (tar and tar!=" ") else None
            if (tar and tar!=" ") and st.checkbox("train_test_split") :
                para_ts=st.number_input("test_size",min_value=0.0,value=0.25,max_value=1.00)
                para_trs=st.number_input("train_size",min_value=0.0,value=(1-para_ts),max_value=(1-para_ts))
                para_rs_str=st.text_input("random_state",value=None)
                para_rs=(int(para_rs_str) if (para_rs_str!=None and para_rs_str.isnumeric()) else None)
                para_s=st.selectbox("shuffle",[True,False])
                
                X_train, X_test, Y_train, Y_test = train_test_split(df[dep], df[tar], test_size=para_ts,train_size=para_trs,random_state=para_rs,shuffle=para_s)
                tts=st.info(f"X_train: {X_train.shape},  X_test: {X_test.shape},  Y_train: {Y_train.shape},  Y_test: {Y_test.shape}")
            

            #Linear Regression
            if choose_algo==supervised_models_list[1] and (dep and tar!=" ") :
                obj=models.linear_reg
                linear_reg_para=st.beta_expander("Model Parameters")
                para_fi=linear_reg_para.selectbox("fit_intercept",[True,False])
                para_norm=linear_reg_para.selectbox("normalize",[False,True])
                para_cx=linear_reg_para.selectbox("copy_X",[True,False])
                para_nj_str=linear_reg_para.text_input("n_jobs",value=None)
                para_nj=(int(para_nj_str) if (para_nj_str!=None and para_nj_str.isnumeric()) else None)
                
                obj.regressor=(LinearRegression(fit_intercept=para_fi,normalize=para_norm,copy_X=para_cx,n_jobs=para_nj)) if not pkl_model else pkl_model
                '''\n'''
                try:
                    #print(X_train, X_test, Y_train, Y_test)
                    st.info(f"{obj.regressor.fit(X_train,Y_train) if not pkl_model else pkl_model }")
                    
                except:
                    e=sys.exc_info()
                    st.error(e)
                #print(X_test)
                
                m=obj.model(X_test ,Y_test ) if tts else obj.model(X_train ,Y_train )
                

                if m[0]=='success':
                    st.markdown("<h3>Attributes :</h3>",unsafe_allow_html=True)
                    st.write("Score")
                    st.success(m[1])
                    st.write("Coef_")
                    st.success(list(m[2]))
                    st.write("Intercept_")
                    st.success(m[3])

                    st.markdown("<h3> Metrics :</h3>",unsafe_allow_html=True)
                    st.write("Mean_Absolute_Error")
                    st.success(m[4])
                    st.write("Mean_Squared_Error")
                    st.success(m[5])
                    st.write("Root_Mean_Squared_Error")
                    st.success(m[6])

                    st.markdown("<h3> Prediction :</h3>",unsafe_allow_html=True)
                    txt_empt=st.empty()
                    random_val=st.checkbox("Select Random Values")
                    
                    txt_ip=txt_empt.text_input("Enter Value")
                    if (txt_ip =="")  and random_val:
                        temp_df=list(df[dep].sample(n=1).values[0])
                        txt_ip=(",".join(str(x) for x in temp_df))
                        txt_empt.text_input("Enter Value",value=txt_ip)
                    obj.predict(txt_ip)
                else:
                    st.error("\n\n".join(str(x) for x in m))
            
            
            #Logistic Regression
                 
            elif choose_algo==supervised_models_list[2] and (dep and tar!=" ") :
                obj=models.logistic_reg
                logistic_reg_para=st.beta_expander("Model Parameters")
                para_fi=logistic_reg_para.selectbox("fit_intercept",[True,False])
                para_is=logistic_reg_para.number_input("intercept_scaling",value=1)
                para_rs_str=logistic_reg_para.text_input("random_state",value=None)
                para_rs=(int(para_rs_str) if (para_rs_str!=None and para_rs_str.isnumeric()) else None)
                para_mi=logistic_reg_para.number_input("max_iter",value=100)
                para_nj_str=logistic_reg_para.text_input("n_jobs",value=None)
                para_nj=(int(para_nj_str) if (para_nj_str!=None and para_nj_str.isnumeric()) else None)
                obj.log_regressor=(LogisticRegression(fit_intercept=para_fi, intercept_scaling=para_is,random_state=para_rs,max_iter=para_mi,n_jobs=para_nj)) if not pkl_model else pkl_model
                '''\n'''
                try:
                    st.info(f"{obj.log_regressor.fit(X_train,Y_train) if not pkl_model else pkl_model }")
                except:
                    e=sys.exc_info()
                    st.error(e)
                m=obj.model(X_test ,Y_test ) if tts else obj.model(X_train ,Y_train ) 
                
                if m[0]=='success':
                    st.markdown("<h3>Attributes :</h3>",unsafe_allow_html=True)
                    st.write("Score")
                    st.success(m[1])
                    st.write("Coef_")
                    st.success(list(*m[2]))
                    st.write("Intercept_")
                    st.success(*m[3])
                    
                    st.markdown("<h3> Metrics :</h3>",unsafe_allow_html=True)
                    st.write("Classification_Report")
                    st.dataframe(pd.DataFrame(m[4]).transpose())
                    

                    st.markdown("<h3> Prediction :</h3>",unsafe_allow_html=True)
                    txt_empt=st.empty()
                    random_val=st.checkbox("Select Random Values")
                    txt_ip=txt_empt.text_input("Enter Value")
                    if (txt_ip =="")  and random_val:
                        temp_df=list(df[dep].sample(n=1).values[0])
                        txt_ip=(",".join(str(x) for x in temp_df))
                        txt_empt.text_input("Enter Value",value=txt_ip)
                    obj.predict(txt_ip)                
                            
                else:
                    st.error("\n\n".join(str(x) for x in m))
            
            #navie bayes
            elif choose_algo==supervised_models_list[3]and (dep and tar!=" "):
                
                obj=models.navie_bayes
                obj.naviebayes=(GaussianNB())if not pkl_model else pkl_model 
                '''\n'''
                try:
                    st.info(f"{obj.naviebayes.fit(X_train,Y_train) if not pkl_model else pkl_model }")
                except:
                    e=sys.exc_info()
                    st.error(e)
                m=obj.model(X_test ,Y_test ) if tts else obj.model(X_train ,Y_train )


                if m[0]=='success':
                    st.markdown("<h3> Metrics :</h3>",unsafe_allow_html=True)
                    st.write("Accuracy_Score")
                    st.success(m[1])
                    st.write("Classification_Report")
                    st.dataframe(pd.DataFrame(m[2]).transpose())

                    st.markdown("<h3> Prediction :</h3>",unsafe_allow_html=True)
                    txt_empt=st.empty()
                    random_val=st.checkbox("Select Random Values")
                    txt_ip=txt_empt.text_input("Enter Value")
                    if (txt_ip =="")  and random_val:
                        temp_df=list(df[dep].sample(n=1).values[0])
                        txt_ip=(",".join(str(x) for x in temp_df))
                        txt_empt.text_input("Enter Value",value=txt_ip)
                    obj.predict(txt_ip)            
                else:
                    st.error("\n\n".join(str(x) for x in m))
            
            #Knn
            elif choose_algo==supervised_models_list[4]and (dep and tar!=" "):
                obj=models.knclassifier
                knn_para=st.beta_expander("Model Parameters")
                para_n=knn_para.slider("n_neighors",min_value=1,value=5)
                para_w=knn_para.selectbox("weights",['uniform','distance'])
                para_a=knn_para.selectbox("algorithm",['auto','ball_tree','kd_tree','brute'])
                para_nj_str=knn_para.text_input("n_jobs",value=None)
                para_nj=(int(para_nj_str) if (para_nj_str!=None and para_nj_str.isnumeric()) else None)
                obj.knn=(KNeighborsClassifier(n_neighbors=int(para_n),weights=para_w,algorithm=para_a,n_jobs=para_nj))if not pkl_model else pkl_model 
                '''\n'''
                try:
                    st.info(f"{obj.knn.fit(X_train,Y_train) if not pkl_model else pkl_model }")
                except:
                    e=sys.exc_info()
                    st.error(e)
                m=obj.model(X_test ,Y_test ) if tts else obj.model(X_train ,Y_train )
                
                if m[0]=='success':
                    st.markdown("<h3>Metrics :</h3>",unsafe_allow_html=True)
                    st.write("Classification_Report")
                    st.dataframe(pd.DataFrame(m[1]).transpose())
                    
                    st.markdown("<h3> Prediction :</h3>",unsafe_allow_html=True)
                    txt_empt=st.empty()
                    random_val=st.checkbox("Select Random Values")
                    txt_ip=txt_empt.text_input("Enter Value")
                    if (txt_ip =="")  and random_val:
                        temp_df=list(df[dep].sample(n=1).values[0])
                        txt_ip=(",".join(str(x) for x in temp_df))
                        txt_empt.text_input("Enter Value",value=txt_ip)
                    obj.predict(txt_ip)           
                else:
                    st.error("\n\n".join(str(x) for x in m))
            
            #decision tree classifier
            elif choose_algo==supervised_models_list[5]and (dep and tar!=" "):
                obj=models.decisiontree
                dtc_para=st.beta_expander("Model Parameters")                
                para_c=dtc_para.selectbox("criterion",['gini','entropy'])
                para_s=dtc_para.selectbox("splitter",['best','random'])
                para_rs_str=dtc_para.text_input("random_state",value=None)
                para_rs=(int(para_rs_str) if (para_rs_str!=None and para_rs_str.isnumeric()) else None)
                obj.dtc=(DecisionTreeClassifier(criterion=para_c,splitter=para_s,random_state=para_rs))if not pkl_model else pkl_model 
                '''\n'''
                try:
                    st.info(f"{obj.dtc.fit(X_train,Y_train) if not pkl_model else pkl_model }")
                except:
                    e=sys.exc_info()
                    st.error(e)
                m=obj.model(X_test ,Y_test ) if tts else obj.model(X_train ,Y_train )
                
                if m[0]=='success':
                    st.markdown("<h3>Metrics :</h3>",unsafe_allow_html=True)
                    st.write("Classification_Report")
                    st.dataframe(pd.DataFrame(m[1]).transpose())
                    
                    st.markdown("<h3> Prediction :</h3>",unsafe_allow_html=True)
                    txt_empt=st.empty()
                    random_val=st.checkbox("Select Random Values")
                    txt_ip=txt_empt.text_input("Enter Value")
                    if (txt_ip =="")  and random_val:
                        temp_df=list(df[dep].sample(n=1).values[0])
                        txt_ip=(",".join(str(x) for x in temp_df))
                        txt_empt.text_input("Enter Value",value=txt_ip)
                    obj.predict(txt_ip)           
                else:
                    st.error("\n\n".join(str(x) for x in m))
            

            #support vector machine classifier
            elif choose_algo==supervised_models_list[6]and (dep and tar!=" "):
                obj=models.supportvectormachineclassifier
                svc_para=st.beta_expander("Model Parameters")
                para_c=svc_para.number_input("C",min_value=0.0,value=1.0)
                para_d=svc_para.number_input("degree",min_value=1,value=3)
                para_g=svc_para.selectbox("gamma",['scale','auto'])
                para_cw=svc_para.selectbox("class_weight",[None,dict(),'balanced'])
                obj.svc=(SVC(C=para_c,degree=para_d,gamma=para_g,class_weight=para_cw))if not pkl_model else pkl_model 
                '''\n'''
                try:
                    st.info(f"{obj.svc.fit(X_train,Y_train) if not pkl_model else pkl_model }")
                except:
                    e=sys.exc_info()
                    st.error(e)
                m=obj.model(X_test ,Y_test ) if tts else obj.model(X_train ,Y_train )
                
                if m[0]=='success':
                    st.markdown("<h3>Metrics :</h3>",unsafe_allow_html=True)
                    st.write("Classification_Report")
                    st.dataframe(pd.DataFrame(m[1]).transpose())
                    
                    st.markdown("<h3> Prediction :</h3>",unsafe_allow_html=True)
                    txt_empt=st.empty()
                    random_val=st.checkbox("Select Random Values")
                    txt_ip=txt_empt.text_input("Enter Value")
                    if (txt_ip =="")  and random_val:
                        temp_df=list(df[dep].sample(n=1).values[0])
                        txt_ip=(",".join(str(x) for x in temp_df))
                        txt_empt.text_input("Enter Value",value=txt_ip)
                    obj.predict(txt_ip)           
                else:
                    st.error("\n\n".join(str(x) for x in m))

            #kmeans clustering
            elif choose_algo==un_supervised_models_list[7]and (dep):
                obj=models.kmeansclustering
                kmc_para=st.beta_expander("Model Parameters")
                para_nc=kmc_para.number_input("n_clusters",min_value=1,value=8)
                para_rs_str=kmc_para.text_input("random_state",value=None)
                para_rs=(int(para_rs_str) if (para_rs_str!=None and para_rs_str.isnumeric()) else None)
                para_init=kmc_para.selectbox("init",["k-means++",'random'])
                obj.kmeans=(KMeans(n_clusters=para_nc,random_state=para_rs,init=para_init))if not pkl_model else pkl_model 
                '''\n'''
                try:
                    st.info(f"{obj.kmeans.fit(X_train) if not pkl_model else pkl_model }")
                except:
                    e=sys.exc_info()
                    st.error(e)
                m=obj.model(X_train )
                
                if m[0]=='success':
                    st.markdown("<h3>Attributes :</h3>",unsafe_allow_html=True)
                    st.write("Cluster Centers_ ")
                    st.write(m[1])
                    st.write("Labels_ ")
                    st.write(m[2])

                    
                    st.markdown("<h3> Prediction :</h3>",unsafe_allow_html=True)
                    txt_empt=st.empty()
                    random_val=st.checkbox("Select Random Values")
                    txt_ip=txt_empt.text_input("Enter Value")
                    if (txt_ip =="")  and random_val:
                        temp_df=list(df[dep].sample(n=1).values[0])
                        txt_ip=(",".join(str(x) for x in temp_df))
                        txt_empt.text_input("Enter Value",value=txt_ip)
                    obj.predict(txt_ip)           
                else:
                    st.error("\n\n".join(str(x) for x in m))
            
            #principal component analysis
            elif choose_algo==un_supervised_models_list[8]and (dep):
                obj=models.principalcomponentanalysis
                pca_para=st.beta_expander("Model Parameters")
                para_nc_str=pca_para.text_input("n_components",value=None)
                para_nc=(int(para_nc_str) if (para_nc_str!=None and para_nc_str.isnumeric()) else None)
                para_c=pca_para.selectbox("copy",[True,False])
                para_svd=pca_para.selectbox("svd_solver",['auto','full','arpack','randomized'])
                para_rs_str=pca_para.text_input("random_state",value=None)
                para_rs=(int(para_rs_str) if (para_rs_str!=None and para_rs_str.isnumeric()) else None)
                
                obj.pca=(PCA(n_components=para_nc,copy=para_c,svd_solver=para_svd,random_state=para_rs))if not pkl_model else pkl_model 
                '''\n'''
                try:
                    st.info(f"{obj.pca.fit(X_train) if not pkl_model else pkl_model }")
                except:
                    e=sys.exc_info()
                    st.error(e)
                m=obj.model(X_train ) 
                
                if m[0]=='success':
                    st.markdown("<h3>Attributes :</h3>",unsafe_allow_html=True)
                    st.write("Components_")
                    st.write(m[1])
                    st.write("Explained Variance_")
                    st.write(m[2])
                    st.write("Explained Variance Ratio_")
                    st.write(m[3])
                    st.write("Singular Values_")
                    st.write(m[4])

                    st.markdown("<h3> Transform :</h3>",unsafe_allow_html=True)
                   
                    st.write(m[5])
                    st.write("Shape: ",m[5].shape)
                         
                else:
                    st.error("\n\n".join(str(x) for x in m))
            
        if m[0]=='success' and st.checkbox("Build Pickle Model"):
            st.markdown(models.download_button(choose_algo) , unsafe_allow_html=True)
    except:pass
    
    
