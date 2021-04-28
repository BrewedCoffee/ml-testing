import csv
import requests
import os
import io
import sklearn.preprocessing

import pandas as pd

DATA_DIR = 'data'

def remove(original, element):
    copy = original.copy()
    copy.remove(element)
    return copy

# Note: use instances to hold data instead of classes
# so not all data is loaded when data.py is loaded

class Data:
    URL = None
    COLUMN_LABELS = None

    @staticmethod
    def encode_columns(data, *args):
        encoder = sklearn.preprocessing.LabelEncoder()
        data_copy = data.copy()
        for column_name in args:
            encoder.fit(data_copy[column_name])
            data_copy[column_name] = encoder.transform(data_copy[column_name])
        return data_copy

    @classmethod
    def encode(cls, data):
        raise NotImplementedError

    @classmethod
    def get_raw(cls):
        os.makedirs(DATA_DIR, exist_ok=True)
        data_file_path = f'{DATA_DIR}/{cls.__name__}.csv'
        if os.path.isfile(data_file_path):
            return pd.read_csv(data_file_path, names=cls.COLUMN_LABELS)
        csv_string = requests.get(cls.URL).content.decode()
        with open(data_file_path, 'w') as data_file:
            data_file.write(csv_string)
        return pd.read_csv(io.StringIO(csv_string), names=cls.COLUMN_LABELS)

    def __init__(self):
        self.raw = self.get_raw()
        self.encoded = self.encode(self.raw)

class ClassificationData(Data):
    CLASSES = None
    DEPENDENT_COLUMN = 'class'

    @classmethod
    def get_independent(cls, data):
        return data[remove(cls.COLUMN_LABELS, cls.DEPENDENT_COLUMN)]

    @classmethod
    def get_dependent(cls, data):
        return data[cls.DEPENDENT_COLUMN]

    def __init__(self):
        super().__init__()

class IrisData(ClassificationData):
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    COLUMN_LABELS = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    CLASSES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    def __init__(self):
        super().__init__()

    @classmethod
    def encode(cls, data):
        return Data.encode_columns(data, 'class')

class BreastCancerData(ClassificationData):
    URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases'
           '/breast-cancer/breast-cancer.data')
    COLUMN_LABELS = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes',
                     'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    CLASSES = ['no-recurrence-events', 'recurrence-events']

    def __init__(self):
        super().__init__()

    @classmethod
    def encode(cls, data):
        return Data.encode_columns(data, *cls.COLUMN_LABELS)
