from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Membaca dataset
dataset = pd.read_csv('dataset.csv',delimiter=";")

# Membagi data atribut dan label
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

# Membangun model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Menerima input dari form
    gula_darah = float(request.form['gula_darah'])
    kolesterol = float(request.form['kolesterol'])
    asam_urat = float(request.form['asam_urat'])

    # Melakukan prediksi menggunakan model KNN
    hasil_prediksi = knn.predict([[gula_darah, kolesterol, asam_urat]])

    # Menentukan pesan hasil prediksi
    pesan = 'Sehat' if hasil_prediksi[0] == 'sehat' else 'Sakit'

    return render_template('index.html', hasil_prediksi=pesan)

if __name__ == '__main__':
    app.run()
