import numpy as np
from tensorflow.keras.models import load_model
import pickle

# load model
model = load_model("./model/model.keras")

with open('./model/X_scaler.pkl', 'rb') as f1:
   X_scaler = pickle.load(f1)
   
with open('./model/Y_scaler.pkl', 'rb') as f2:
   Y_scaler = pickle.load(f2)

with open('./model/ocp_encoder.pkl', 'rb') as f3:
    ocp_encoder = pickle.load(f3)

def prepare_input(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity):
    # encode
    ocp_encoded = ocp_encoder.transform(np.array([[ocean_proximity.lower()]])) # array(1, 5)

    X_remain = np.array([[longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income]]) # array(1, 8)
    X_predict = np.concatenate((X_remain, ocp_encoded), axis=1)  # array(1, 13)

    # scale
    X_predict = X_scaler.transform(X_predict)
    return X_predict

input = {
    "longitude": -121.22,
    "latitude": 39.43,
    "housing_median_age": 17.0,
    "total_rooms": 2254.0,
    "total_bedrooms": 485.0,
    "population": 1007.0,
    "households": 433.0,
    "median_income": 1.7000,
    "ocean_proximity": "INLAND"
    # "median_house_value": 92300.0
}

X_pred = prepare_input(**input)
Y_pred = model.predict(X_pred)
Y_pred = Y_scaler.inverse_transform(Y_pred)

print(Y_pred)