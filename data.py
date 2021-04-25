import requests
import os
import io

import pandas as pd

IRIS_URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
IRIS_COLUMN_NAMES = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']
IRIS_CLASSES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

class Data:
    def __init__(self):
        self.iris = Data.get_iris()

    @staticmethod
    def get_iris():
        csv_data = io.StringIO(requests.get(IRIS_URL).content.decode())
        df = pd.read_csv(csv_data, names=IRIS_COLUMN_NAMES)
        return df
