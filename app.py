import numpy as np
import pandas as pd
from flask import Flask, request, render_template  
import pickle


flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))  


@flask_app.route('/')
def Home():
    return render_template('sample.html')

#prediction of model using machine learning 

@flask_app.route('/predict', methods=['GET','POST'])
def predict():

        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)
        
        return render_template("sample.html", prediction_text = "The recommended crop is{}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)