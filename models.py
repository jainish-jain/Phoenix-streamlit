import numpy as np
import pandas as pd
import pickle,random,base64,tempfile,json
from pandas.core import base
#import seaborn as sns
from streamlit.errors import Error 
import sys,os,uuid,re,time
import streamlit as st
from sklearn import metrics , preprocessing
from sklearn.metrics import classification_report,mean_absolute_error,mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split
import sklearn

#Models
from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



def label_encoder(data_encode,col_names):
    le=preprocessing.LabelEncoder()
    temp_df=pd.DataFrame(columns=col_names)

    return le.fit_transform(data_encode)

# @st.cache()
def download_button(choose_algo, pickle_it=False):
    """
    download_filename, button_text,
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    
    # to delete older files 1hr to clear memory
    try:
        #print(os.getcwd())
        for f in os.listdir("Model/"):
            #print(f,((time.time() - os.path.getctime(os.path.join("Model/",f)))/3600))
            if ((time.time() - os.path.getctime(os.path.join("Model/",f)))/3600)>1 and f!="temp":
                os.remove(os.path.join("Model/", f))
    except:pass
    
    
    supervised_models_list={1:'Linear Regression',2:'Logistic Regression',3:'Naïve Bayes: GaussianNB',4:'K-Nearest Neighbours',5:'Decision Tree Classifier',6:'Support Vector Machine Classifier'}
    un_supervised_models_list={7:"K Means Clustering",8:"Principal Component Analysis"}

    if choose_algo==supervised_models_list[1]:
        download_filename=str(linear_reg.download())
    elif choose_algo==supervised_models_list[2]:
        download_filename=str(logistic_reg.download())
    elif choose_algo==supervised_models_list[3]:
        download_filename=str(navie_bayes.download())    
    elif choose_algo==supervised_models_list[4]:
        download_filename=str(knclassifier.download())
    elif choose_algo==supervised_models_list[5]:
        download_filename=str(decisiontree.download())
    elif choose_algo==supervised_models_list[6]:
        download_filename=str(supportvectormachineclassifier.download())
    elif choose_algo==un_supervised_models_list[7]:
        download_filename=str(kmeansclustering.download())
    elif choose_algo==un_supervised_models_list[8]:
        download_filename=str(principalcomponentanalysis.download())

    #print(download_filename)
    with open(download_filename, 'rb') as f:
            object_to_download = f.read()
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.30em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
                font-size: inherit;

            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">Download Model</a><br></br>'
    return dl_link


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)



class linear_reg:
    regressor=LinearRegression()
    def model(X,Y):
        # X=df[dep]
        # Y=df[tar]#.values.reshape(-1, 1)
        try:
            #linear_reg.regressor.fit(X,Y)
            reg_predict=linear_reg.regressor.predict(X)
            reg_score= linear_reg.regressor.score(X,Y)
            reg_coef=linear_reg.regressor.coef_
            reg_intercept=linear_reg.regressor.intercept_
            
            reg_mae=mean_absolute_error(Y,reg_predict)
            reg_mse=mean_squared_error(Y,reg_predict)
            reg_rmse=np.sqrt(mean_squared_error(Y, reg_predict))
        except:
            e = sys.exc_info()
            return e
        return ("success",reg_score,reg_coef,reg_intercept,reg_mae,reg_mse,reg_rmse)
    
    def predict(txt_ip):
        #txt_ip=st.text_input("Enter Value","")
        if st.button("Predict") and txt_ip!="":
            #st.balloons()
            try:
                predict_ip=list(map(float,str(txt_ip).split(",")))
                predict_text= linear_reg.regressor.predict([predict_ip])
                st.success(*predict_text)
            except:
                e=sys.exc_info()
                st.error(e)
    def download():
        Pkl_Filename = "Model/Linear_Regression_"+str(random.randint(10**5,10**10))+str(".pkl")
        with open(Pkl_Filename, 'wb') as file:
            pickle.dump(linear_reg.regressor, file)
            os.chmod(Pkl_Filename, 0o777)
        return Pkl_Filename


class logistic_reg:
    log_regressor=LogisticRegression()
    def model(X,Y):
        
        try:
            #logistic_reg.log_regressor.fit(X,Y)
            log_score=logistic_reg.log_regressor.score(X,Y)  
            log_predict=logistic_reg.log_regressor.predict(X)
            log_reg= classification_report(Y,log_predict,output_dict=True)         
            log_reg_coef=logistic_reg.log_regressor.coef_
            log_reg_intercept=logistic_reg.log_regressor.intercept_
            
        except:
            e = sys.exc_info()
            return e
        return ("success",log_score,log_reg_coef,log_reg_intercept,log_reg)
    
    def predict(txt_ip):
        if   st.button("Predict") and txt_ip!="" :
            try:
                predict_ip=list(map(float,str(txt_ip).split(",")))
                predict_text= logistic_reg.log_regressor.predict([predict_ip])
                st.success(*predict_text)
                predict_text_proba=logistic_reg.log_regressor.predict_proba([predict_ip])
                st.write("Predict_Proba")
                st.success(*predict_text_proba)
            except:
                e=sys.exc_info()
                st.error(e)
    def download():
        Pkl_Filename = "Model/Logistic_Regression_"+str(random.randint(10**5,10**10))+str(".pkl")
        with open(Pkl_Filename, 'wb') as file:
            pickle.dump(logistic_reg.log_regressor, file)
            os.chmod(Pkl_Filename, 0o777)
        return Pkl_Filename
        

        
    

class navie_bayes:
    naviebayes=GaussianNB()
    
    def model(X,Y):
        try:
            #navie_bayes.naviebayes.fit(X,Y)
            navie_bayes_predict= navie_bayes.naviebayes.predict(X)
            nav_as= accuracy_score(Y,navie_bayes_predict)
            nav_cr= classification_report(Y,navie_bayes_predict,output_dict=True)
        except:
            e = sys.exc_info()
            return e
        return ("success",nav_as,nav_cr)
    
    def predict(txt_ip):
        #txt_ip=st.text_input("Enter Value","")
        if st.button("Predict") and txt_ip!="":
            #st.balloons()
            try:
                predict_ip=list(map(float,str(txt_ip).split(",")))
                predict_text= navie_bayes.naviebayes.predict([predict_ip])
                st.success(*predict_text)
            except:
                e=sys.exc_info()
                st.error(e)

    def download():
        Pkl_Filename = "Model/Navie_Bayes_"+str(random.randint(10**5,10**10))+str(".pkl")
        with open(Pkl_Filename, 'wb') as file:
            pickle.dump(navie_bayes.naviebayes, file)
            os.chmod(Pkl_Filename, 0o777)
        return Pkl_Filename


class knclassifier:
    knn=KNeighborsClassifier()
    
    def model(X,Y):
        try:
            #knclassifier.knn.fit(X,Y)
            knn_predict=knclassifier.knn.predict(X)
            knn_cr= classification_report(Y,knn_predict,output_dict=True)
            # knn_score= knclassifier.knn.score(X,Y)
            
        except:
            e = sys.exc_info()
            return e
        return ("success",knn_cr)
    
    def predict(txt_ip):
        #txt_ip=st.text_input("Enter Value","")
        if st.button("Predict") and txt_ip!="":
            #st.balloons()
            try:
                predict_ip=list(map(float,str(txt_ip).split(",")))
                predict_text= knclassifier.knn.predict([predict_ip])
                st.success(*predict_text)
                st.write("Predict_Proba")
                predict_text_proba=knclassifier.knn.predict_proba([predict_ip])
                st.success(*predict_text_proba)
            except:
                e=sys.exc_info()
                st.error(e)
    def download():
        Pkl_Filename = "Model/K-Nearest_Neighbours_"+str(random.randint(10**5,10**10))+str(".pkl")

        with open(Pkl_Filename, 'wb') as file:
            pickle.dump(knclassifier.knn, file)
            os.chmod(Pkl_Filename, 0o777)
        return Pkl_Filename

class decisiontree:
    dtc=DecisionTreeClassifier()
    
    def model(X,Y):
        try:
            #decisiontree.dtc.fit(X,Y)
            dtc_predict=decisiontree.dtc.predict(X)
            dtc_cr= classification_report(Y,dtc_predict,output_dict=True)

        except:
            e=sys.exc_info()
            st.error(e)

        return ("success",dtc_cr)
    
    def predict(txt_ip):
        #txt_ip=st.text_input("Enter Value","")
        if st.button("Predict") and txt_ip!="":
            #st.balloons()
            try:
                predict_ip=list(map(float,str(txt_ip).split(",")))
                predict_text= decisiontree.dtc.predict([predict_ip])
                st.success(*predict_text)
            except:
                e=sys.exc_info()
                st.error(e)
    def download():
        Pkl_Filename = "Model/Decision_Tree_Classifier_"+str(random.randint(10**5,10**10))+str(".pkl")

        with open(Pkl_Filename, 'wb') as file:
            pickle.dump(decisiontree.dtc, file)
            os.chmod(Pkl_Filename, 0o777)
        return Pkl_Filename


class supportvectormachineclassifier:
    svc=SVC()
    
    def model(X,Y):
        try:
            #supportvectormachineclassifier.svc.fit(X,Y)
            svc_predict=supportvectormachineclassifier.svc.predict(X)
            svc_cr= classification_report(Y,svc_predict,output_dict=True)

        except:
            e=sys.exc_info()
            st.error(e)

        return ("success",svc_cr)
    
    def predict(txt_ip):
        #txt_ip=st.text_input("Enter Value","")
        if st.button("Predict") and txt_ip!="":
            #st.balloons()
            try:
                predict_ip=list(map(float,str(txt_ip).split(",")))
                predict_text= supportvectormachineclassifier.svc.predict([predict_ip])
                st.success(*predict_text)
            except:
                e=sys.exc_info()
                st.error(e)
    def download():
        Pkl_Filename = "Model/Support_Vector_Machine_Classifier_"+str(random.randint(10**5,10**10))+str(".pkl")

        with open(Pkl_Filename, 'wb') as file:
            pickle.dump(supportvectormachineclassifier.svc, file)
            os.chmod(Pkl_Filename, 0o777)
        return Pkl_Filename

class kmeansclustering:
    kmeans=KMeans()
    def model(X):
        try:
            #kmeans_predict=kmeansclustering.kmeans.predict(X)
            kmeans_cen=kmeansclustering.kmeans.cluster_centers_
            kmeans_label=kmeansclustering.kmeans.labels_
        except:
            e=sys.exc_info()
            st.error(e)

        return ("success",kmeans_cen,kmeans_label)
    
    def predict(txt_ip):
        #txt_ip=st.text_input("Enter Value","")
        if st.button("Predict") and txt_ip!="":
            #st.balloons()
            try:
                predict_ip=list(map(float,str(txt_ip).split(",")))
                predict_text= kmeansclustering.kmeans.predict([predict_ip])
                st.success(*predict_text)
            except:
                e=sys.exc_info()
                st.error(e)
    def download():
        Pkl_Filename = "Model/K_Means_Clustering_"+str(random.randint(10**5,10**10))+str(".pkl")

        with open(Pkl_Filename, 'wb') as file:
            pickle.dump(kmeansclustering.kmeans, file)
            os.chmod(Pkl_Filename, 0o777)
        return Pkl_Filename

class principalcomponentanalysis:
    pca=PCA()
    def model(X):
        try:
            #kmeans_predict=kmeansclustering.kmeans.predict(X)
            pca_c=principalcomponentanalysis.pca.components_
            pca_ev=principalcomponentanalysis.pca.explained_variance_
            pca_evr=principalcomponentanalysis.pca.explained_variance_ratio_
            pca_sv=principalcomponentanalysis.pca.singular_values_

            pca_transform=principalcomponentanalysis.pca.transform(X)
            
        except:
            e=sys.exc_info()
            st.error(e)

        return ("success",pca_c,pca_ev,pca_evr,pca_sv,pca_transform)
        
    def download():
        Pkl_Filename = "Model/Principal_Component_Analysis_"+str(random.randint(10**5,10**10))+str(".pkl")

        with open(Pkl_Filename, 'wb') as file:
            pickle.dump(principalcomponentanalysis.pca, file)
            os.chmod(Pkl_Filename, 0o777)
        return Pkl_Filename