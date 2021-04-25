import requests
import os

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
