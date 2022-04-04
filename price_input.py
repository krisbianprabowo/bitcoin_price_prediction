import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

#------------------------------load model-----------------------------------#
with(open("scale_pipe.pkl", "rb")) as file:
    preprocess = pickle.load(file)

model = tf.keras.models.load_model('lstm_24.h5')

#--------------------------create input form------------------------------------#
uploaded= 0
uploaded_file = st.file_uploader('Upload Historical Data BTC/USDT', type=['csv'])
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    st.success('File uploaded!')
    df = pd.read_csv(uploaded_file, header=None)
    uploaded=1
st.write('Atau jika tidak memiliki Data yang memadai, silahkan klik tombol dibawah')
if st.button('Get Dataset'):
    df = pd.read_csv('inference/BTCUSDT-1h-2022-02.csv', header=None)
    uploaded=1

close = st.number_input('Close Price (in $)', 0.00, value=44000.00, step=0.01)

#--------------------------predict------------------------------------#

sequence_length = 50

def partition_dataset_inf(sequence_length, data):
    x = []
    data_len = data.shape[0]

    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) 
    
    x = np.array(x)
    return x

if uploaded==1:
    row_inf = 7 * 24
    df_selected = df[:row_inf].iloc[:, 4]
    df_selected[len(df_selected)+1] = close
    df_scaled = preprocess.transform(df_selected.values[len(df_selected)-50:].reshape(-1,1))
    df_scaled = df_scaled.reshape(-1, 50, 1)

    y_pred = model.predict(df_scaled)
    y_pred = preprocess.inverse_transform(y_pred.reshape(-1, 1))
st.subheader('Prediksi Harga Penutupan:')
if uploaded==1:
    st.subheader(f"${y_pred[0][0]}")
    # Visualization of the First Three Data
    df_real = preprocess.inverse_transform(df_scaled.reshape(1,-1))
    def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$price$"):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(series, ".-", color="#E91D9E")
        if y is not None:
            plt.plot(50, y, "bx", markersize=10, color="blue")
        if y_pred is not None:
            plt.plot(50, y_pred, "ro")
        plt.grid(True)
        if x_label:
            plt.xlabel(x_label, fontsize=16,)
        if y_label:
            plt.ylabel(y_label, fontsize=16)
        st.pyplot(fig)

    plot_series(df_real.reshape(-1), y_pred[0][0])
else:
    st.subheader("--")

