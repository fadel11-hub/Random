import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load
from streamlit_option_menu import option_menu

# Sidebar menu
with st.sidebar:
    selected = option_menu("Menu", ["Prediksi", "Settings"])

# Load scaler dan model
scaler_x = load('scaler_x.pkl')
scaler_y = load('scaler_y.pkl')
model = load_model('model (1).h5')

# Halaman Home
if selected == "Prediksi":
    st.title('Project MSIB SCADA - Prediksi Daya Turbin Angin')

    # Input pengguna
    wind_speed = st.number_input("Masukkan kecepatan angin (m/s)", min_value=0.0, step=0.1)
    wind_direction = st.number_input("Masukkan arah angin (derajat)", min_value=0.0, max_value=360.0, step=0.1)

    # Tombol Prediksi
    if st.button("Prediksi Daya Turbin"):
        try:
            # Data input
            input_array = np.array([[wind_speed, wind_direction]])
            input_array = scaler_x.transform(input_array)
            input_array = input_array.reshape((1, 1, input_array.shape[1]))  # LSTM input

            # Prediksi
            predictions = model.predict(input_array)
            predictions_denorm = scaler_y.inverse_transform(predictions)
            prediction_result = float(predictions_denorm[0][0])

            st.success(f"Perkiraan daya turbin: {prediction_result:.2f} kW")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")

# Halaman Settings
elif selected == "Settings":
    st.title("Pengaturan Aplikasi")
    st.write("Halaman ini bisa digunakan untuk mengatur parameter aplikasi di masa depan.")

