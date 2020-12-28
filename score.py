import json
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('capstone_hyperdrive_best_model')
    model = joblib.load(model_path)
    print(model)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model.predict(data)
    return json.dumps(y_hat.tolist())