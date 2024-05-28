import streamlit as st
import pandas as pd
import numpy as np 
import tempfile
from bokeh.plotting import figure
from itertools import combinations
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split

def download_model(model):
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        model.save(tmp.name)
        tmp.seek(0)
        st.download_button(label='Download Model', data=tmp.read(), file_name='model.h5',mime='application/octet-stream')

def train_test(f_data,o_data,test_data_size):
    X_train, X_test, y_train, y_test = train_test_split(f_data,o_data,random_state=42,test_size=test_data_size)
    return X_train, X_test, y_train, y_test

def build_reg_model(data):
    model_r = Sequential()
    model_r.add(Dense(1028, input_dim=data.shape[1], activation='relu'))
    model_r.add(Dense(512,activation='relu'))
    model_r.add(Dropout(0.2))
    model_r.add(Dense(512,activation='relu'))
    model_r.add(Dense(256,activation='relu'))
    model_r.add(Dense(256,activation='relu'))
    model_r.add(Dropout(0.2))
    model_r.add(Dense(128,activation='relu'))
    model_r.add(Dense(128,activation='relu'))
    model_r.add(Dense(64, activation='relu'))
    model_r.add(Dropout(0.2))
    model_r.add(Dense(128,activation='relu'))
    model_r.add(Dense(128,activation='relu'))
    model_r.add(Dense(64, activation='relu'))
    model_r.add(Dropout(0.2))
    model_r.add(Dense(64,activation='relu'))
    model_r.add(Dense(32,activation='relu'))
    model_r.add(Dense(1, activation='relu'))
    return model_r

def build_class_model(data):
    model_c = Sequential()
    model_c.add(Dense(128, input_dim=data.shape[1], activation='relu'))
    model_c.add(Dense(512,activation='relu'))
    model_c.add(Dropout(0.2))
    model_c.add(Dense(512,activation='relu'))
    model_c.add(Dense(256,activation='relu'))
    model_c.add(Dense(256,activation='relu'))
    model_c.add(Dropout(0.2))
    model_c.add(Dense(128,activation='relu'))
    model_c.add(Dense(128,activation='relu'))
    model_c.add(Dense(64, activation='relu'))
    model_c.add(Dropout(0.2))
    model_c.add(Dense(64,activation='relu'))
    model_c.add(Dense(32,activation='relu'))
    model_c.add(Dense(64, activation='relu'))
    model_c.add(Dropout(0.2))
    model_c.add(Dense(64,activation='relu'))
    model_c.add(Dense(64, activation='relu'))
    model_c.add(Dense(1, activation='sigmoid'))
    return model_c

def analyze_csv(data):
    is_numeric = data.iloc[:,-1].astype(str).str.isnumeric().all()
    num_unique_values = data.iloc[:,-1].nunique()
    if is_numeric:
        if num_unique_values == 2:
            return True
        else :
            return False

def main():
    st.set_page_config(page_title="Tool for Analysis of MD Data",page_icon="chart_with_upwards_trend",layout="wide")
    st.markdown("# Model Building Web Page")
    st.divider()
    features = st.file_uploader("Upload the csv file which consists of Features", type=['csv'])
    outcome_vector = st.file_uploader("Upload the csv file which consists of Outcome Vector", type=['csv'])
    train_data_size = st.number_input("Train Data Split", min_value=0.0, max_value=1.0, step=0.01,placeholder="Type a number...")    
    test_data_size = st.number_input("Test Data Split", min_value=0.0, max_value=1.0, step=0.01,placeholder="Type a number...")
    validation_data = st.number_input("Validation Data Split", min_value=0.0,max_value=0.9, step=0.01,placeholder="Type a number...")
    epochs = st.number_input("Number of Epochs", min_value=1, step=1,placeholder="Type a number...")
    if (train_data_size + test_data_size + validation_data == 1):
        button = st.button("Upload")
    else:
        total = train_data_size + test_data_size + validation_data 
        st.write(total)
        st.stop()
    if button:
        if features and outcome_vector:
            st.info("Successfully uploaded the files",icon="ℹ️")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f_csv:
                f_csv.write(features.getvalue())
                f_csv_path = f_csv.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as o_csv:
                o_csv.write(outcome_vector.getvalue())
                o_csv_path = o_csv.name
            
            f_data = pd.read_csv(f_csv_path)
            o_data = pd.read_csv(o_csv_path)

            if(o_data.shape[1] == 1 and o_data.shape[0] == f_data.shape[0]):
                st.info("Successfully validated the feature file and outcome vector file",icon="ℹ️")
                problem = analyze_csv(o_data)
                if not problem:
                    st.info("The Problem is a Regression Problem",icon="ℹ️")
                    with st.spinner("We started building the model....."):            
                        model = build_reg_model(f_data)
                        st.info("Model is built Successfully",icon="ℹ️")
                        model.compile(loss = 'mean_squared_error',
                                        optimizer = 'adam',
                                        metrics = ['mse','mae'])
                        st.info("Model is compiled Successfully",icon="ℹ️")
                        st.info("Splitting the data into Train and Test",icon="ℹ️") 
                        X_train, X_test, Y_train, Y_test = train_test(f_data,o_data,test_data_size)
                        st.info("Successfully performed the Train_test_split ",icon="ℹ️")
                        st.write(X_train.shape)
                        st.write(X_test.shape)
                        st.write(Y_train.shape)
                        st.write(Y_test.shape)
                        st.warning("Started training the model",icon="⚠️")
                        history = model.fit(X_train, Y_train, epochs=epochs,validation_split=validation_data)
                        st.info("Succesfully trained the model",icon="ℹ️")
                        st.write("Plotting the MSE Metric....")
                        epoch = np.arange(epochs)
                        p = figure(
                            title='Epochs vs MSE',
                            x_axis_label='Epochs',
                            y_axis_label='MSE')
                        p.line(epoch,history.history['mse'], legend_label='MSE ', line_width=2)
                        st.bokeh_chart(p, use_container_width=True)
                        st.write("Plotting the MAE Metric....")
                        p = figure(
                            title='Epochs vs MAE',
                            x_axis_label='Epochs',
                            y_axis_label='MAE')
                        p.line(epoch,history.history['mae'], legend_label='MAE ', line_width=2)
                        st.bokeh_chart(p, use_container_width=True)
                        st.info("Evaluating the model using test data",icon="ℹ️")
                        loss, mae, mse  = model.evaluate(X_test,Y_test)
                        st.write("Test Loss: ",loss)
                        st.write("Mean Squared Error: ",mse)
                        st.write("Mean Absolute Error: ",mae)
                        download_model(model)
                    st.stop()
                else:
                    st.info("The Problem is a Binary Classification Problem",icon="ℹ️")
                    with st.spinner("We Started Building the Model ...."):
                        model = build_class_model(f_data)
                        st.success("Model is built Successfully")
                        model.compile(loss = 'binary_crossentropy',
                                        optimizer = 'adam',
                                        metrics = ['accuracy'])
                        st.success("Model is compiled Successfully")
                        st.info("Splitting the data into Train and Test",icon="ℹ️") 
                        X_train, X_test, Y_train, Y_test = train_test(f_data,o_data,test_data_size)
                        st.info("Successfully performed the Train_test_split operation",icon="ℹ️")
                        st.write(X_train.shape)
                        st.write(X_test.shape)
                        st.write(Y_train.shape)
                        st.write(Y_test.shape)
                        st.warning("Started training the model",icon="⚠️")
                        history = model.fit(X_train, Y_train, epochs=epochs,validation_split=validation_data)
                        st.info("Succesfully trained the model",icon="ℹ️")
                        st.write("Plotting the Accuracy....")
                        epoch = np.arange(epochs)
                        p = figure(
                            title='Epochs vs Accuracy',
                            x_axis_label='Epochs',
                            y_axis_label='Accuracy',)
                        p.line(epoch,history.history['accuracy'], legend_label='Accuracy', line_width=2)
                        st.bokeh_chart(p, use_container_width=True)
                        st.info("Evaluating the model using test data",icon="ℹ️")
                        test_loss, test_accuracy = model.evaluate(X_test,Y_test)
                        st.write("Test Loss: ",test_loss)
                        st.write("Test Accuracy: ",test_accuracy)
                        download_model(model)
                    st.stop()
            else:
                st.warning("Please Upload the correct outcome Vector File with only 1 column",icon="⚠️")
                st.stop()
        else:
            st.warning("Please Upload the files",icon="⚠️")
            st.stop()

if __name__ == '__main__':
    main()