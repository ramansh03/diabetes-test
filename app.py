from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np
import sklearn
import pickle


file='rf_diabetes.pkl'
model=pickle.load(open(file,'rb'))


app = Flask(__name__, template_folder='templates')

@app.route("/")
@app.route("/Diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route('/predict', methods = ["GET","POST"])
def predict():
    if(request.method=='POST'):
        preg = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        bp = int(request.form['BloodPressure'])
        bmi = int(request.form['BMI'])
        dpf = int(request.form['DiabetesPedigreeFunction'])
        age = float(request.form['Age'])
       
        
        data = np.array([[preg, glucose, bp, bmi, dpf, age]])
        my_prediction = model.predict(data)
        return render_template('result.html', prediction=my_prediction)

if __name__ == "__main__":
    app.run(debug=True)
