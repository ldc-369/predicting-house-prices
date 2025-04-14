from flask import Flask,request,render_template
from keras.models import load_model
import  pickle
import numpy as np

app = Flask(__name__)

# load model
model = load_model("./predicting-house-prices/model/model.h5")

with open('./predicting-house-prices/model/scaler_F.pkl', 'rb') as f1:
   scaler_F = pickle.load(f1)
   
with open('./predicting-house-prices/model/scaler_y.pkl', 'rb') as f2:
   scaler_y = pickle.load(f2)

with open('./predicting-house-prices/model/onehot_encoder.pkl', 'rb') as f3:
    onehot_encoder = pickle.load(f3)

def prepare_input(longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity):
    # encoding
    ocp_encoded = onehot_encoder.transform(np.array([[ocean_proximity]]))  # (1, 5)

    features_remain = np.array([[longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income]])  # (1, 8)

    # nối mảng
    F_predict = np.concatenate((features_remain, ocp_encoded), axis=1)  # (1, 13)

    # scaling
    F_predict = scaler_F.transform(F_predict)

    return F_predict

# creating routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/house-price", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        longitude = request.form['longitude']
        latitude = request.form['latitude']
        houseage = request.form['houseage']
        houserooms = request.form['houserooms']
        totlabedrooms = request.form['totlabedrooms']
        population = request.form['population']
        households = request.form['households']
        medianincome = request.form['medianincome']
        oceanproximity = request.form['oceanproximity']
        
        F_predict = prepare_input(longitude,latitude,houseage,houserooms,totlabedrooms,population,households,medianincome,oceanproximity)
        
        # predict
        y_predict = model.predict(F_predict)
        house_price = scaler_y.inverse_transform(y_predict)[0][0]

        return render_template('index.html', house_price = house_price)


if __name__ == "__main__":
    app.run(debug=True)