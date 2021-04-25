import requests

import pandas as pd

IRIS_URL = ('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/'
            'iris/iris.data')

def get_data():
    response = requests.get(IRIS_URL)

def read_data():
    df = pd.read_csv('iris.csv')
    print(df.to_dict())

read_data()