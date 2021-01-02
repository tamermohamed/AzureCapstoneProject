import pickle
import json
import numpy as np
import joblib
import pandas
from azureml.core.model import Model


def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('capstone_hyperdrive_best_model')
    model = joblib.load(model_path)


def run(raw_data):
    data = json.loads(raw_data)['data']

    input_data = pandas.DataFrame.from_dict(data)
    # make prediction

    y_hat = model.predict(input_data)
    return json.dumps(y_hat.tolist())
