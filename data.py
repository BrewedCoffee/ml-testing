import requests
import os

import pandas as pd

IRIS_URL = ('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/'
            'iris/iris.data')
DATA_PATH = 'data/'
IRIS_PATH = DATA_PATH + 'iris.csv'

def get_iris():
    if os.path.isfile(IRIS_PATH):
        return
    response = requests.get(IRIS_URL)
    os.makedirs(DATA_PATH)
    with open(IRIS_PATH, 'wb') as f:
        f.write(response.content)

def serve_data(headers=[]):
    names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']
    df = pd.read_csv('iris.csv',names=names)
    return df[headers]

# print(serve_data(['Class']))
