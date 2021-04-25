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
<<<<<<< HEAD
    return pd.read_csv('iris.csv', names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
=======
    df = pd.read_csv(IRIS_PATH)
    return df
    print(df.to_dict())

read_data()
>>>>>>> 4e26b1381d6fb47e74616d625d3788ac5110eeea
