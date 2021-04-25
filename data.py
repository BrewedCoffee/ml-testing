import requests
import os
import io

import pandas as pd

DATA_PATH = 'data/'

IRIS_URL = ('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/'
            'iris/iris.data')
IRIS_NAMES = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']

class Data:
    def __init__(self):
        self.data = get_iris()

    def get_iris():
        csv_data = io.StringIO(requests.get(IRIS_URL).content.decode())
        df = pd.read_csv(csv_data, names=IRIS_NAMES)
        return df
