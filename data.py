import requests

IRIS_URL = ('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/'
            'iris/iris.data')

def get_data():
    response = requests.get(IRIS_URL)