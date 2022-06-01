from copyreg import pickle
from flask import Flask, render_template, request
from regex import R
from sqlalchemy import true
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        inputedData = [request.form["BMI"], request.form["Smoke_or_not"], request.form["alochol_or_not"], request.form["Stroke"], request.form["Physical_health"], request.form["Mental_health"], request.form["Diff_walking"], request.form["sex"],
                       request.form["age"], request.form["race"], request.form["Diabetic"], request.form["Physical_activity"], request.form["General_health"], request.form["Sleep_time"], request.form["Asthma"], request.form["Kidney_disease"], request.form["skin_cancer"]]
    inputedData = np.array([inputedData])
   # print("data-----------------------:", inputedData)
    inputedData = inputedData.reshape(1, -1)
    prediction = model.predict_proba(inputedData)
    prediction = prediction[0][1]

    return render_template("index.html", n=prediction)


if __name__ == "__main__":
    app.run(debug=true)
