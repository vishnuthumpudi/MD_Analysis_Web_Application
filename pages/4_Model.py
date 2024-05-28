import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

def make_predictions(data):
    with st.spinner("Predicting...."):
        loaded_model = load_model('model1.h5') #Load the Model
        st.success('Model loaded successfully') # Displays Success message
        scaler = StandardScaler()
        test_x = scaler.fit_transform(data) # Scaling the data
        predictions = loaded_model.predict(test_x) # Making Predictions
    st.success('Predictions made successfully') # Displays Success message
    return predictions # Returns predictions

def main():
    st.set_page_config(page_title='Model', page_icon=':bar_chart:', layout='wide')
    st.header("This is the Model Page")
    st.divider()
    st.header("Want to know more about Adapatability")
    lst = ["It refers to a measure of the per-atom conformational plasticity of the protein.",
           "It is a metric used to assess how individual atoms within a protein structure are likely to adapt or change their positions in response to ligand binding during molecular dynamics simulations.",
           "The adaptability model aims to identify elements of the biomolecule structure that are flexible or rigid, providing insights into the dynamic behavior of the protein in the presence of ligands."]
    s = ""
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
    st.divider()
    csv_file = st.file_uploader('Upload CSV file', type=['csv'])
    first_column = st.checkbox('Ignore First Column')
    button = st.button('Predict')

    if button:
        if csv_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                tmp_csv.write(csv_file.getvalue())
                csv_path = tmp_csv.name
            if first_column:
                try:
                    data = pd.read_csv(csv_path)
                    st.success('Data loaded successfully')
                    data = data.iloc[:, 1:]
                    predictions = make_predictions(data)
                    data['Predicted_Adaptabilty'] = predictions
                    st.write("Here is your final DataFrame")
                    st.dataframe(data,use_container_width=True)
                    st.text("- If Adaptability == 0, indicates a neutral or average level of Adaptability (neither the molecule is highly flexible nor rigid")
                    st.text("- If Adaptability ~= 1, indicates a high flexibility and adaptability of the atoms and also indicating significant conformational changes")
                    st.text("- If Adaptability ~= -1, indicates a high rigidity and low adaptability of the atoms and also indicating minimum or no significant conformational changes")
                    st.toast("Hurrah! Predictions made successfully", icon="🎉")
                    st.stop()
                except Exception as e:
                    st.error(e,icon="🚨")
                    st.stop()
            try:
                data = pd.read_csv(csv_path)
                st.success('Data loaded successfully')
                predictions = make_predictions(data)
                data['Predicted_Adaptabilty'] = predictions
                st.write("Here is your final DataFrame")
                st.dataframe(data,use_container_width=True)
                st.text("If Adaptability == 0, indicates a neutral or average level of Adaptability (neither the molecule is highly flexible nor rigid")
                st.text("If Adaptability ~= 1, indicates a high flexibility and adaptability of the atoms and also indicating significant conformational changes")
                st.text("If Adaptability ~= -1, indicates a high rigidity and low adaptability of the atoms and also indicating minimum or no significant conformational changes")    
                st.toast("Hurrah! Predictions made successfully", icon="🎉")
            except Exception as e:
                st.error(e,icon="🚨")
                st.stop()

if __name__ == '__main__':
    main()