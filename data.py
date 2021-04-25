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

def read_data():
    df = pd.read_csv(IRIS_PATH)
    return df
    print(df.to_dict())

read_data()
