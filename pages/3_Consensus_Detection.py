import os
import sys
import time
import streamlit as st
import numpy as np
import pandas as pd
import tempfile

# Main Function
def main():
    st.set_page_config(page_title="Consensus Outlier Detection", page_icon=":shark:", layout="wide")
    st.title("Consensus Outlier Detection")
    st.divider()
    csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
    consensus_threshold = st.number_input("Consensus Threshold", min_value=5, max_value=12, step=1,placeholder="Type a number...")
    first_column = st.checkbox("Ignore First Column")
    button = st.button("Process")
    if button:
        if csv_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                tmp_csv.write(csv_file.getvalue())
                csv_path = tmp_csv.name
            if first_column:
                try:
                    data = np.genfromtxt(csv_path, delimiter=',',skip_header=1,usecols=range(1, 13))
                    row_sums = np.sum(data,axis = 1)
                    count = 0
                    frames_list = []

                    for i,row_sums in enumerate(row_sums):
                        if(row_sums >= consensus_threshold):
                            count+=1
                            frames_list.append(i)
                    if frames_list == []:
                        st.write("No Outliers detected in the trajectory")
                        st.stop()
                    else:
                        st.write(f"Number of frames that are outliers in the trajectory are : {count}")
                        st.write(f"Frames that are outliers in the trajectory are : {frames_list}")
                    st.stop()
                except Exception as e:
                    st.warning("Some error encountered",icon="🚨")
                    st.stop()
            try:
                data = np.genfromtxt(csv_path, delimiter=',',skip_header=1)
                row_sums = np.sum(data,axis = 1)
                count = 0
                frames_list = []

                for i,row_sums in enumerate(row_sums):
                    if(row_sums >= consensus_threshold):
                        count+=1
                        frames_list.append(i)
                if frames_list == []:
                    st.write("No Outliers detected in the trajectory")
                    st.stop()
                else:
                    st.write(f"Number of frames that are outliers in the trajectory are : {count}")
                    st.write(f"Frames that are outliers in the trajectory are : {frames_list}")
                st.stop()
            except Exception as e:
                st.error(e,icon="🚨")
                st.stop()

#Driver Program         
if __name__ == '__main__':
    main()