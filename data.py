import csv
import requests
import os
import io
import zipfile
import sklearn.preprocessing
import shutil
import datetime

import pandas as pd

import utils

DATA_DIR = 'data'

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
        data_file_path = f'{DATA_DIR}/{cls.__name__}.csv'
        if os.path.isfile(data_file_path):
            return pd.read_csv(data_file_path, names=cls.COLUMN_LABELS)
        csv_string = requests.get(cls.URL).content.decode()
        os.makedirs(DATA_DIR, exist_ok=True)
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
    def independent(cls, data):
        return data[utils.remove_element(cls.COLUMN_LABELS, cls.DEPENDENT_COLUMN)]

    @classmethod
    def dependent(cls, data):
        return data[cls.DEPENDENT_COLUMN]

    def __init__(self):
        super().__init__()

class AirQualityData(Data):
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'
    COLUMN_LABELS = ['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
                    'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
                    'PT08.S5(O3)', 'T', 'RH', 'AH']

    @classmethod
    def get_raw(cls):
        data_file_path = f'{DATA_DIR}/{cls.__name__}.csv'
        if os.path.isfile(data_file_path):
            return pd.read_csv(data_file_path, parse_dates=[0])
        os.makedirs(DATA_DIR, exist_ok=True)
        response = requests.get(cls.URL)
        with open('temp.zip', 'wb') as zip_file:
            zip_file.write(response.content)
        with zipfile.ZipFile('temp.zip') as zip_file:
            zip_file.extractall('tempdir')
        dataframe = pd.read_excel(
            'tempdir/AirQualityUCI.xlsx', skiprows=0, names=cls.COLUMN_LABELS,
            usecols=cls.COLUMN_LABELS, parse_dates={'timestamp': [0, 1]})
        dataframe.to_csv(f'{DATA_DIR}/{cls.__name__}.csv', index=False)
        os.remove('temp.zip')
        shutil.rmtree('tempdir')
        return dataframe

    @classmethod
    def encode(cls, data):
        return data

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
