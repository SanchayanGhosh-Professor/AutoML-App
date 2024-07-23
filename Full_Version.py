import streamlit as st
import pandas as pd
import pickle
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup as class_setup, pull as class_pull, compare_models as class_compare_models, save_model as class_save_model
from pycaret.regression import setup as reg_setup, pull as reg_pull, compare_models as reg_compare_models, save_model as reg_save_model
import os


with st.sidebar:
    st.image("_Pngtree_future_technology_artificial_intelligence_ai_5750901-removebg-preview.png")
    st.title("AutoML App (Prototype)")
    choices = st.radio("Navigation", ["Upload","EDA","Machine Learning","Pipeline Download"])
    st.info("This application allows you to build a automated pipeline.")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv",index_col=None)

if choices == "Upload":
    st.title("Upload your Data for Modelling")
    file = st.file_uploader("Upload your Dataset Here")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)


if choices == "EDA":
    st.title("Automated EDA of the Data")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)

for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            # Handle non-convertible columns appropriately, if needed
            pass

if choices == "Machine Learning":
    st.title("Machine Learning")
    algorithm_type = st.selectbox("Select Your Algorithm", ["Classification","Regression"])
    if algorithm_type == "Classification":
        target = st.selectbox("Select Your Target", df.columns)
        if st.button("Train Model"):
            class_setup(data = df,target=target,verbose=False,use_gpu=False)
            setup_df = class_pull()
            st.info("This is the ML Experiment Settings")
            st.dataframe(setup_df)
            best_model = class_compare_models()
            compare_df = class_pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            best_model
            class_save_model(best_model,"best_model")
    elif algorithm_type == "Regression":
        target = st.selectbox("Select Your Target", df.columns)
        if st.button("Train Model"):
            reg_setup(data = df,target=target,verbose=False,use_gpu=False)
            setup_df = reg_pull()
            st.info("This is the ML Experiment Settings")
            st.dataframe(setup_df)
            best_model = reg_compare_models()
            compare_df = reg_pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            best_model
            reg_save_model(best_model,"best_model")    


if choices == "Pipeline Download":
    with open("best_model.pkl","rb") as f:
        st.download_button("Download the file",f,"trained_model.pkl")