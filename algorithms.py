from scipy import stats
from matplotlib import pyplot
import numpy
import pandas

from sklearn import preprocessing
import sklearn

import data

class Algorithm:
    data = data.Data()

    @classmethod
    def on_iris(cls):
        raise NotImplementedError

class Classifier(Algorithm):
    @staticmethod
    def encode_labels(data, col):
        category_encoder = preprocessing.LabelEncoder()
        data_copy = data.copy()
        if isinstance(data, pandas.DataFrame):
            category_encoder.fit(data[col])
            data_copy[col] = category_encoder.transform(data[col])
        elif isinstance(data, numpy.ndarray):
            category_encoder.fit(data[:, col])
            data_copy[:, col] = category_encoder.transform(data[:, col])
        return data_copy

    @classmethod
    def train(cls):
        raise NotImplementedError

    @classmethod
    def classify(cls):
        raise NotImplementedError

class LinearRegression(Algorithm):
    @classmethod
    def on_iris(cls):
        return cls.fit(
            cls.data.iris['Sepal Length'].to_list(),
            cls.data.iris['Sepal Width'].to_list())

    @staticmethod
    def fit(x, y):
        result = stats.linregress(x, y)
        print(f"R-squared: {result.rvalue**2:.6f}")
        pyplot.plot(x, y, 'o', label='original data')
        for i in range(len(y)):
            y[i] = result.intercept + result.slope*x[i]
        pyplot.plot(x, y, 'r', label='fitted line')
        pyplot.show()

class LogisticRegression(Classifier):
    @classmethod
    def on_iris(cls):
        iris_classes = (data.IRIS_CLASSES[0], data.IRIS_CLASSES[1])
        iris_dataframe = cls.data.iris.loc[cls.data.iris['Class'].isin(iris_classes)]
        iris_dataframe = Classifier.encode_labels(iris_dataframe, 'Class')
        iris_data_array = iris_dataframe.to_numpy()
        iris_data_array = Classifier.encode_labels(iris_data_array, -1)
        print(iris_data_array)

    @staticmethod
    def train(x, y):
        print(x, y)

    @staticmethod
    def classify(model, x):
        pass

LogisticRegression.on_iris()

LinearRegression.on_iris()
