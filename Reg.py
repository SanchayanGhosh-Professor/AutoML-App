import streamlit as st
import pandas as pd
import pickle
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import setup,pull,compare_models,save_model
import os


with st.sidebar:
    st.image("_Pngtree_future_technology_artificial_intelligence_ai_5750901-removebg-preview.png")
    st.title('AutoML App')
    choices = st.radio("Navigation", ["Upload","Profiling","ML","Download"])
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


if choices == "Profiling":
    st.title("Automated EDA of the Data")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)


if choices == "ML":
    st.title("Machine Learning")
    target = st.selectbox("Select Your Target", df.columns)
    if st.button("Train Model"):
        setup(data = df,target=target,verbose=False,use_gpu=True)
        setup_df = pull()
        st.info("This is the ML Experiment Settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,"best_model")

if choices == "Download":
    with open("best_model.pkl","rb") as f:
        st.download_button("Download the file",f,"trained_model.pkl")