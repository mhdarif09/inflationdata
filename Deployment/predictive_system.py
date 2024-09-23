import streamlit as st
import pandas as pd
import sklearn
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
# model_path = r"C:/Users/Nick Mathew/OneDrive - Bina Nusantara/Kuliah/c a w u 4/Machine Learning/project/Prediksi-Inflasi-Pendidikan-Indonesia"
# model_rfr = load_mo(model_path + '/pendidikan_model.pkl')
labelEncoder = LabelEncoder()
sc = StandardScaler()

# Define the array of items
items = ['KOTA MEULABOH', 'KOTA BANDA ACEH', 'KOTA LHOKSEUMAWE',
         'KOTA SIBOLGA', 'KOTA PEMATANG SIANTAR', 'KOTA MEDAN',
         'KOTA PADANGSIDIMPUAN', 'KOTA GUNUNGSITOLI', 'KOTA PADANG',
         'KOTA BUKITTINGGI', 'TEMBILAHAN', 'KOTA PEKANBARU', 'KOTA DUMAI',
         'BUNGO', 'KOTA JAMBI', 'KOTA PALEMBANG', 'KOTA LUBUKLINGGAU',
         'KOTA BENGKULU', 'KOTA BANDAR LAMPUNG', 'KOTA METRO',
         'TANJUNG PANDAN', 'KOTA PANGKAL PINANG', 'KOTA BATAM',
         'KOTA TANJUNG PINANG', 'DKI JAKARTA', 'KOTA BOGOR',
         'KOTA SUKABUMI', 'KOTA BANDUNG', 'KOTA CIREBON', 'KOTA BEKASI',
         'KOTA DEPOK', 'KOTA TASIKMALAYA', 'CILACAP', 'PURWOKERTO', 'KUDUS',
         'KOTA SURAKARTA', 'KOTA SEMARANG', 'KOTA TEGAL', 'KOTA YOGYAKARTA',
         'JEMBER', 'BANYUWANGI', 'SUMENEP', 'KOTA KEDIRI', 'KOTA MALANG',
         'KOTA PROBOLINGGO', 'KOTA MADIUN', 'KOTA SURABAYA',
         'KOTA TANGERANG', 'KOTA CILEGON', 'KOTA SERANG', 'SINGARAJA',
         'KOTA DENPASAR', 'KOTA MATARAM', 'KOTA BIMA', 'WAINGAPU',
         'MAUMERE', 'KOTA KUPANG', 'SINTANG', 'KOTA PONTIANAK',
         'KOTA SINGKAWANG', 'SAMPIT', 'KOTA PALANGKA RAYA', 'KOTA BARU',
         'TANJUNG', 'KOTA BANJARMASIN', 'KOTA BALIKPAPAN', 'KOTA SAMARINDA',
         'TANJUNG SELOR', 'KOTA TARAKAN', 'KOTA MANADO', 'KOTA KOTAMOBAGU',
         'LUWUK', 'KOTA PALU', 'BULUKUMBA', 'WATAMPONE', 'KOTA MAKASSAR',
         'KOTA PARE-PARE', 'KOTA PALOPO', 'KOTA KENDARI', 'KOTA BUA-BAU',
         'KOTA GORONTALO', 'MAMUJU', 'KOTA AMBON', 'KOTA TUAL',
         'KOTA TERNATE', 'MANOKWARI', 'KOTA SORONG', 'MERAUKE', 'TIMIKA',
         'KOTA JAYAPURA', 'INDONESIA']

# Display the selected item


def main():
    st.title('Prediksi Inflasi Pendidikan Indonesia')
    
    kota = st.selectbox('Pilih Kota: ', items)
    year = st.selectbox('Pilih tahun yang akan diprediksi (2024 s/d 2025)', range(2024, 2026))
    months = st.selectbox('Pilih bulan yang akan diprediksi (Januari = 1 s/d Desember = 12)', range(1, 13))
    category = st.selectbox('Pilih kategori inflasi yang akan diprediksi', ['Pendidikan', 'Menengah', 'Tinggi', 'Lainnya'])
    print(months, kota, year, category)

    edu_predict = ''
    if (st.button('Prediksi')):
        edu_predict = predict(kota, year, months, category)
        st.success('Prediksi berhasil dilakukan!')
        st.write(f'Prediksi Inflasi Pendidikan: \n', edu_predict)
        st.write('Kota: ', kota)
        st.write('Tahun: ', year)
        st.write('Bulan: ', months)
        st.write('Kategori: ', category)

        

def category_selection(category):
    if (category=='Pendidikan'):
        return 'model_pendidikan.pkl'
    elif(category=='Menengah'):
        return 'model_menengah.pkl'
    elif(category=='Tinggi'):
        return 'model_tinggi.pkl'
    return 'model_lainnya.pkl'

def predict(kota, year, months, category):
    import joblib
    model = joblib.load(category_selection(category))
    
    # Encode categorical variables
    kota_encoded = labelEncoder.fit_transform([kota])[0]
    category_encoded = labelEncoder.fit_transform([category])[0]
    
    # Scale numerical variables
    year = np.array(year).reshape(-1, 1)
    year_scaled = sc.fit_transform(year)
    
    # Create data array
    data = np.array([[kota_encoded, *year_scaled.flatten(), months, category_encoded]])
    
    # Make prediction
    prediction = model.predict(data)

    if prediction[0] == 0:
        return f"Inflation Rate: {prediction[0]} (Inflasi Pendidikan Tidak Naik)"
    else:
        return f"Inflation Rate {prediction[0]} (Inflasi Pendidikan Naik)"

if __name__ == '__main__':
    main()
