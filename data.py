import requests
import os

import pandas as pd

IRIS_URL = ('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/'
            'iris/iris.data')
DATA_PATH = 'data/'

def get_iris():
    IRIS_PATH = DATA_PATH + 'iris.csv'
    if os.path.isfile(IRIS_PATH):
        return
    response = requests.get(IRIS_URL)
    os.makedirs(DATA_PATH)
    with open(IRIS_PATH, 'wb') as f:
        f.write(response.content)

def read_data():
    return pd.read_csv('iris.csv', names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])