import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras import layers, models, callbacks
from keras.models import load_model
model = load_model("CNN_model.keras")

import pickle

with open("label_encoder.pkl", "rb") as f:
    labelencoder = pickle.load(f)


# segment = X_test[1233]
# test_data = segment.reshape(1, _features, 1).astype('float32')

### How to use labelencoder
pred = model.predict(test_data)
predicted_class = np.argmax(pred)
original_label = labelencoder.inverse_transform([predicted_class])[0]
print("Predicted class label:", original_label)
original_real_label = labelencoder.inverse_transform([np.argmax(Y_test[1233])])[0]
print(original_real_label)