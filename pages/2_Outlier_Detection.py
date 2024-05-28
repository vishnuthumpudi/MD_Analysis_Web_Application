import os
import sys
import pyod
import tempfile
import numpy as np
import pandas as pd
from time import time
import streamlit as st
from scipy import stats
from numpy import percentile
from pyod.models.ecod import ECOD
from pyod.models.abod import ABOD 
from pyod.models.hbos import HBOS 
from pyod.models.knn import KNN 
from pyod.models.lof import LOF 
from pyod.models.mcd import MCD 
from pyod.models.ocsvm import OCSVM 
from pyod.models.pca import PCA
from pyod.models.lmdd import LMDD 
from sklearn.cluster import DBSCAN

def fitting_classifiers(outliers_fraction=0.01):
    st.write(outliers_fraction)
    random_state = np.random.RandomState(42)
    classifiers = {
        'Angle-based Outlier Detector (ABOD)':
            ABOD(contamination=outliers_fraction),
        'Histogram-base Outlier Detection (HBOS)': HBOS(
            contamination=outliers_fraction),
        'K Nearest Neighbors (KNN)': KNN(
            contamination=outliers_fraction),
        'Local Outlier Factor (LOF)':
            LOF(n_neighbors=35, contamination=outliers_fraction),
        'Minimum Covariance Determinant (MCD)': MCD(
            contamination=outliers_fraction, random_state=random_state),
        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
        'Principal Component Analysis (PCA)': PCA(
            contamination=outliers_fraction, random_state=random_state),
        'LMDD': LMDD(contamination=outliers_fraction),
        'Extended Connectivity-Based Outlier Factor (ECOD)': ECOD(contamination=outliers_fraction),
        'DBSCAN' : 1,
        'Z-Score' : 1,
        'IQR' : 1
    }
    for i, clf in enumerate(classifiers.keys()):
        st.write('Model', i + 1, clf)

def detect_ecod(data,outliers_fraction):
    clf = ECOD(contamination=outliers_fraction)
    clf.fit(data)
    y_pred = clf.labels_
    return y_pred

def detect_abod(data,outliers_fraction):
    clf = ABOD(contamination=outliers_fraction)
    clf.fit(data)
    y_pred = clf.labels_
    return y_pred

def detect_hbos(data,outliers_fraction):
    clf = HBOS(contamination=outliers_fraction)
    clf.fit(data)
    y_pred = clf.labels_
    return y_pred

def detect_knn(data,outliers_fraction):
    clf = KNN(contamination=outliers_fraction)
    clf.fit(data)
    y_pred = clf.labels_
    return y_pred

def detect_lof(data,outliers_fraction):
    clf = LOF(n_neighbors=35, contamination=outliers_fraction)
    clf.fit(data)
    y_pred = clf.labels_
    return y_pred

def detect_mcd(data,outliers_fraction):
    clf = MCD(contamination=outliers_fraction, random_state=42)
    clf.fit(data)
    y_pred = clf.labels_
    return y_pred

def detect_ocsvm(data,outliers_fraction):
    clf = OCSVM(contamination=outliers_fraction)
    clf.fit(data)
    y_pred = clf.labels_
    return y_pred

def detect_pca(data,outliers_fraction):
    clf = PCA(contamination=outliers_fraction, random_state=42)
    clf.fit(data)
    y_pred = clf.labels_
    return y_pred

def detect_lmdd(data,outliers_fraction):
    clf = LMDD(contamination=outliers_fraction)
    clf.fit(data)
    y_pred = clf.labels_
    return y_pred

def detect_dbscan(data):
    clf = DBSCAN(eps=3, min_samples=2)
    clf.fit(data)
    y_pred = clf.labels_
    y_pred_adjusted = [1 if label < 0 else 0 for label in y_pred]
    return y_pred_adjusted

def detect_zscore(data):
    z = data.apply(lambda x: np.abs((x-x.mean())/x.std()))
    z_threshold = 2
    is_outlier = (z > z_threshold).astype(int)
    is_outlier.replace({True: 1, False: 0},inplace=True)
    return is_outlier.any(axis=1).astype(int)

def detect_IQR(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    is_outlier = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1).astype(int)
    return is_outlier.astype(int)

def iqr_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter outliers
    outliers = df[(df < lower_bound) | (df > upper_bound)]
    return outliers

# Function to detect Global Outliers
def detect_outliers(data,outliers_fraction=0.01):
    de = detect_ecod(data,outliers_fraction=0.01)
    da = detect_abod(data,outliers_fraction=0.01)
    dh = detect_hbos(data,outliers_fraction=0.01)
    dk = detect_knn(data,outliers_fraction=0.01)
    dl = detect_lof(data,outliers_fraction=0.01)
    dm = detect_mcd(data,outliers_fraction=0.01)
    do = detect_ocsvm(data,outliers_fraction=0.01)
    dp = detect_pca(data,outliers_fraction=0.01)
    dl = detect_lmdd(data,outliers_fraction=0.01)
    db = detect_dbscan(data)
    dz = detect_zscore(data)
    di = detect_IQR(data)
    fin_df = pd.DataFrame({'ECOD':de,'ABOD':da,'HBOS':dh,'KNN':dk,'LOF':dl,'MCD':dm,'OCSVM':do,'PCA':dp,'LMDD':dl,'DBSCAN':db,'Z-Score':dz,'IQR':di})
    st.success("Successfully detected outliers")
    return fin_df

# Function to detect Outliers Feature by Feature
def detect_Outliers(data,outliers_fraction=0.01):
    dat = data 
    data = data.values.reshape(-1, 1)
    de = detect_ecod(data,outliers_fraction=0.01)
    da = detect_abod(data,outliers_fraction=0.01)
    dh = detect_hbos(data,outliers_fraction=0.01)
    dk = detect_knn(data,outliers_fraction=0.01)
    dl = detect_lof(data,outliers_fraction=0.01)
    dm = detect_mcd(data,outliers_fraction=0.01)
    do = detect_ocsvm(data,outliers_fraction=0.01)
    dp = detect_pca(data,outliers_fraction=0.01)
    dl = detect_lmdd(data,outliers_fraction=0.01)
    db = detect_dbscan(data)
    # dz = detect_zscore(data)
    # di = iqr_outliers(dat)
    fin_df = pd.DataFrame({'ECOD':de,'ABOD':da,'HBOS':dh,'KNN':dk,'LOF':dl,'MCD':dm,'OCSVM':do,'PCA':dp,'LMDD':dl,'DBSCAN':db   })
    return fin_df

# Main Function to detect Outliers
def main():
    st.set_page_config(page_title="Outlier Detection Tool ",page_icon="chart_with_upwards_trend",layout="wide")
    st.markdown("#Outlier Detection web page")
    st.divider()
    csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
    number = st.number_input("Enter the outlier fraction", min_value=0.0, max_value=1.0, step=0.01, placeholder="Type a number...")
    first_col = st.checkbox("Ignore First Column in your CSV file")
    f_b_f = st.checkbox("Compute Outliers Feature By Feature")
    g_o_d = st.checkbox("Compute Outliers Globally")
    button = st.button("Process")
    if button:
        if f_b_f and g_o_d:
            st.warning("Please select only one option from the above",icon="⚠️")
            st.stop();
        if csv_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                tmp_csv.write(csv_file.getvalue())
                csv_path = tmp_csv.name
            if g_o_d: # g_o_d represents global outlier detection
                if first_col:
                    try:
                        data_frame = pd.read_csv(csv_path)
                        df = data_frame.iloc[:,1:]
                        st.success("Successfully loaded the CSV file and we are ready to detect outliers")
                        fitting_classifiers(number)  
                        fin_df = detect_outliers(df,number) 
                        st.dataframe(fin_df,use_container_width=True)
                        st.toast("Hooray! We have detected outliers successfully", icon='🎉')
                        st.stop()
                    except Exception as e:
                        st.error(e,icon="🚨")
                        st.stop()
                try:
                    data_frame = pd.read_csv(csv_path)
                    st.success("Successfully loaded the CSV file and we are ready to detect outliers")
                    fitting_classifiers(number)  
                    fin_df = detect_outliers(data_frame,number) 
                    st.dataframe(fin_df,use_container_width=True)
                    st.toast("Hooray! We have detected outliers successfully", icon='🎉')
                except Exception as e:
                    st.error(e,icon="🚨")
                    st.stop()

            if f_b_f: #f_b_f represents feature by feature
                if first_col:
                    try: 
                        data_frame = pd.read_csv(csv_path)
                        df = data_frame.iloc[:,1:]
                        for col in df.keys():
                            data = df[col]
                            fin_df = detect_Outliers(data,number)
                            st.write("Outliers from column",col)
                            st.dataframe(fin_df,use_container_width=True)
                        st.toast("Hooray! We have detected outliers successfully", icon='🎉')
                        st.stop()
                    except Exception as e:
                        st.error(e,icon="🚨")
                        st.stop()
                try:
                    data_frame = pd.read_csv(csv_path)
                    df = data_frame.iloc[:,1:]
                    for col in df.keys():
                        data = df[col]
                        fin_df = detect_Outliers(data,number)
                        st.write("Outliers from column",col)
                        st.dataframe(fin_df,use_container_width=True)
                    st.toast("Hooray! We have detected outliers successfully", icon='🎉')
                    st.stop()
                except Exception as e:
                    st.error(e,icon="🚨")
                    st.stop()

#Driver Program
if __name__ == '__main__':
    main()