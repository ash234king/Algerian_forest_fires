from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

#import ridge regressor and standard scaler pickle
import os

MODEL_PATH = os.path.join('Algerian_forest_fires','web_app_and_deployment','models', 'ridge.pkl')
SCALER_PATH = os.path.join('Algerian_forest_fires','web_app_and_deployment','models', 'scaler.pkl')

ridge_model = pickle.load(open(MODEL_PATH, 'rb'))
standard_scaler = pickle.load(open(SCALER_PATH, 'rb'))

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/predictData',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data)
        return render_template('result.html',results=result[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")