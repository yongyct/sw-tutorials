#
# Surgical Complications
#
# MLOps III 5-minute tour
#


import pandas as pd
import numpy as np
import joblib


import os
import io
from typing import List, Optional
from scipy.special import expit
g_code_dir = None

###################################
    

def init(code_dir):
    global g_code_dir
    g_code_dir = code_dir


def read_input_data(input_binary_data):
    data = pd.read_csv(io.BytesIO(input_binary_data))    
    return data


def transform(data, model):
    data = data.fillna(0)
    return data


def load_model(code_dir):
    model_path = 'surgical_complications_pipeline.pkl'
    model = joblib.load(os.path.join(code_dir, model_path))
    return model


def score(data, model, **kwargs):
    results = model.predict_proba(data)
    predictions = pd.DataFrame({'0': results[:, 0], '1':results[:, 1]})
    return predictions
