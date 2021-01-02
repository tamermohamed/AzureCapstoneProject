import pickle
import json
import numpy as np
from sklearn.externals import joblib
from azureml.core.model import Model

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('capstone_hyperdrive_best_model')
    model = joblib.load(model_path)

def run(raw_data):

    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model.predict(data)
    return json.dumps(y_hat.tolist())