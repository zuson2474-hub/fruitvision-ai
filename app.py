from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("fruit_model.h5")

# Nama class sesuai dataset
class_names = [
    'apple',
    'banana',
    'grapes',
    'mango',
    'orange'
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return "Tidak ada file"

    file = request.files['image']

    # buka gambar
    img = Image.open(file)

    # ubah ukuran
    img = img.resize((64, 64))

    # convert ke array
    img_array = np.array(img)

    # normalisasi
    img_array = img_array / 255.0

    # tambah dimensi
    img_array = np.expand_dims(img_array, axis=0)

    # prediksi
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    result = class_names[predicted_class]

    confidence = np.max(prediction) * 100

    return render_template(
        'index.html',
        prediction=result,
        confidence=f"{confidence:.2f}%"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
