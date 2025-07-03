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

def prepare_input(input):    
    X_predict = np.empty((0, 13))
    
    for item in input:
        longitude = item["longitude"]
        latitude = item["latitude"]
        housing_median_age = item["housing_median_age"]
        total_rooms = item["total_rooms"]
        total_bedrooms = item["total_bedrooms"]
        population = item["population"]
        households = item["households"]
        median_income = item["median_income"]
        
        # encode
        ocp_encoded = ocp_encoder.transform(np.array([[item["ocean_proximity"].lower()]])) # array(1, 5)

        # scale
        X_remain = np.array([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]]) # array(1, 8)
        X_remain = X_scaler.transform(X_remain)
        
        #concatenate
        X_predict = np.concatenate((X_predict, np.concatenate((X_remain, ocp_encoded), axis=1)), axis=0)  # array(-1, 13)

    return X_predict

def predict(input):
    X_predict = prepare_input(input)
    print(X_predict)
    Y_predict = model.predict(X_predict)

    return Y_scaler.inverse_transform(Y_predict)

input = [
    {
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
    }, 
    {
        "longitude": -122.25,
        "latitude": 37.85,
        "housing_median_age": 52.0,
        "total_rooms": 1627.0,
        "total_bedrooms": 280.0,
        "population": 565.0,
        "households": 259.0,
        "median_income": 3.8462,
        "ocean_proximity": "NEAR BAY",
        # "median_house_value": 342200.0
    },
    {
        "longitude": -122.2300,
        "latitude": 37.8800,
        "housing_median_age": 41.0000,
        "total_rooms": 880.0000,
        "total_bedrooms": 129.0000,
        "population": 322.0000,
        "households": 126.0000,
        "median_income": 8.3252,
        "ocean_proximity": "ISLAND",
        # "median_house_value": 452600.0000
    }
]

Y_predict = predict(input)
print(f"Predicted median_house_value: {Y_predict}")