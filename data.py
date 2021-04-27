import requests
import os
import io
import sklearn.preprocessing

import pandas as pd

def remove(original, element):
    copy = original.copy()
    copy.remove(element)
    return copy

class ClassificationData:
    URL = None
    COLUMN_LABELS = None
    CLASSES = None
    DEPENDENT_COLUMN = None

    @staticmethod
    def encode_columns(data, *args):
        encoder = sklearn.preprocessing.LabelEncoder()
        data_copy = data.copy()
        for column_name in args:
            encoder.fit(data_copy[column_name])
            data_copy[column_name] = encoder.transform(data_copy[column_name])
        return data_copy

    @classmethod
    def get_raw_data(cls):
        response = requests.get(cls.URL).content.decode()
        return pd.read_csv(io.StringIO(response), names=cls.COLUMN_LABELS)

    @classmethod
    def encode(cls, data):
        raise NotImplementedError

    def get_independent(self, data):
        return data[remove(self.__class__.COLUMN_LABELS, self.__class__.DEPENDENT_COLUMN)]

    def get_dependent(self, data):
        return data[self.__class__.DEPENDENT_COLUMN]

    def __init__(self):
        self.raw_data = self.get_raw_data()
        self.encoded_data = self.__class__.encode(self.raw_data)

class IrisData(ClassificationData):
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    COLUMN_LABELS = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    CLASSES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    DEPENDENT_COLUMN = 'class'

    def __init__(self):
        super().__init__()

    @classmethod
    def encode(cls, data):
        return ClassificationData.encode_columns(data, 'class')

class BreastCancerData(ClassificationData):
    URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases'
           '/breast-cancer/breast-cancer.data')
    COLUMN_LABELS = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes',
                     'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    CLASSES = ['no-recurrence-events', 'recurrence-events']
    DEPENDENT_COLUMN = 'class'

    def __init__(self):
        super().__init__()

    @classmethod
    def encode(cls, data):
        return ClassificationData.encode_columns(data, *cls.COLUMN_LABELS)

def main():
    iris = IrisData()
    print(iris.encoded_dependent)

if __name__ == '__main__':
    main()
