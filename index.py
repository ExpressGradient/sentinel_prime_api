from flask import Flask, request
from tensorflow import keras
import numpy as np
import pandas as pd
import math

df = pd.read_csv('./stock_data.csv')
df = df.replace(-1, 0)

ds = df.to_numpy()

train_ds, test_ds = ds[:math.floor(
    0.8 * ds.shape[0])], ds[math.floor(0.8 * ds.shape[0]):]

train_x, train_y = train_ds[:, 0], train_ds[:, 1]

text_vectorizer = keras.layers.TextVectorization(
    ngrams=2,
    max_tokens=10000,
    output_mode='multi_hot'
)

text_vectorizer.adapt(train_x)


app = Flask(__name__)


@app.post('/predict')
def predict():
    text = request.json['text']
    text = text_vectorizer(np.array([text]))

    model = keras.models.load_model('sentinel_prime')
    prediction = model.predict(text)
    return {'prediction': prediction.tolist()}


if __name__ == '__main__':
    app.run(debug=True)
