import json
import numpy as np
import os
# import pickle
import joblib
from sklearn.linear_model import LogisticRegression
 
def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = os.path.join('.', 'outputs', 'hyperdrive_best_model.pkl')
    model = joblib.load(model_path)
 
def run(raw_data):
    data = np.array(json.loads(raw_data))
    # make prediction
    y_hat = model.predict(data)
    return json.dumps(y_hat.tolist())