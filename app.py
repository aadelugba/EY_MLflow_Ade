import numpy as np
from flask import Flask, request, jsonify, render_template
from pickle import load
import joblib
import sys
import mlflow.pyfunc # to load mlflow model
from mlflow.tracking import MlflowClient

# This is just to make the modules discoverable - as these were used to pickle the model
# hence, needed to unpickle it --- otherwise it returns ModuleNotFoundError: when trying to load the model
# sys.path.append(r'model')
app = Flask(__name__)


# HOME ROUTE
@app.route('/')
def home():
    return render_template('home.html')

# PREDICT ROUTE
@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    if request.method == "POST":

        #get form data
        Pregnancies = request.form.get('Pregnancies')
        PlasmaGlucose = request.form.get('PlasmaGlucose')
        DiastolicBloodPressure = request.form.get('DiastolicBloodPressure')
        TricepsThickness = request.form.get('TricepsThickness')
        SerumInsulin = request.form.get('SerumInsulin')
        BMI = request.form.get('BMI')
        DiabetesPedigree = request.form.get('DiabetesPedigree')
        Age = request.form.get('Age')
        
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age)
            # pass prediction to template
            return render_template('predict.html', prediction = prediction)
   
        except ValueError as e:
            # return "Please Enter valid values"
            print(e)
  

def mlflow_model_version():
    client = MlflowClient() # alias for MlflowClient 
    model_name = "Diabetes Prediction Model"
    filter_string = "name='{}'".format(model_name)
    results = client.search_registered_models(filter_string=filter_string)
    for res in results:
        for mv in res.latest_versions:
        # print("name={}; run_id={}; version={}".format(mv.name, mv.run_id, mv.version))
            model_version = mv.version

    return model_name, model_version


def preprocessDataAndPredict(Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age):
    
    #keep all inputs in array
    test_data = [Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age]
    # print(test_data)
    
    #convert value data into numpy array
    test_data = np.array(test_data)
    
    #reshape array - cos predict takes a 2D array--[no_of_obs, no_of_features]
    test_data = test_data.reshape(1,-1)
    # print(test_data)
    
    # Load recent mlflow "registered" model
    # model_name, model_version = mlflow_model_version()
    # model_version_uri = "models:/{model_name}/{model_version}".format(model_name=model_name, model_version=model_version)
    # trained_model = mlflow.pyfunc.load_model(model_version_uri)    # trained_model = joblib.load("output/model.pkl")
    
    # Load non-mlflow model - pickled file on local repo
    # Do not use pickle.load as it gives Pickling error
    trained_model = joblib.load("output_no_mlflow/model.pkl")

    # predict
    prediction = trained_model.predict(test_data)
    
    return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True) # Pointing to Port 80 as I already have mlflow in Port 5000