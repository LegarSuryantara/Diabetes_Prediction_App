import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="SPK Diabetes (KNN)", page_icon="🩺")

try:
    classifier = pickle.load(open('diabetes_model_knn.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except FileNotFoundError:
    st.error("File model tidak ditemukan. Jalankan train_model.py dulu!")
    st.stop()


st.title('🩺 Diagnosa Diabetes (Metode KNN)')
st.markdown("""
Aplikasi ini menggunakan metode **K-Nearest Neighbors (KNN)**. 
Sistem akan membandingkan data Anda dengan 5 data pasien terdekat (paling mirip) di database untuk menentukan hasil.
""")

st.write('---')


col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input('Jumlah Kehamilan', 0, 20, 0)
    Glucose = st.number_input('Glukosa (mg/dL)', 0, 300, 120)
    BloodPressure = st.number_input('Tekanan Darah (mm Hg)', 0, 200, 70)
    SkinThickness = st.number_input('Ketebalan Kulit (mm)', 0, 100, 20)

with col2:
    Insulin = st.number_input('Insulin (mu U/ml)', 0, 900, 79)
    BMI = st.number_input('BMI', 0.0, 100.0, 25.0)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree', 0.0, 3.0, 0.5)
    Age = st.number_input('Umur', 0, 120, 30)


if st.button('Cek Hasil Diagnosa'):
    input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    std_data = scaler.transform(input_data_reshaped)
    
    prediction = classifier.predict(std_data)
    
    st.write('---')
    st.subheader('Hasil Prediksi:')
    
    if (prediction[0] == 0):
        st.success('🟢 Pasien diprediksi **SEHAT (Tidak Diabetes)**.')
    else:
        st.error('🔴 Pasien diprediksi **Menderita DIABETES**.')

# Footer
st.markdown("---")
st.caption("Dikembangkan untuk Tugas SPK | Metode: K-Nearest Neighbors (KNN)")