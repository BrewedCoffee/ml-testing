import scipy.stats as stats
import matplotlib.pyplot as pyplot
import numpy
import pandas
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

from data import Data

class Algorithm:
    data = Data()

    @classmethod
    def on_iris(cls):
        raise NotImplementedError

class Classifier(Algorithm):
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
    @staticmethod
    def train(data):
        print(data)

    @staticmethod
    def classify(model, data):
        pass

LinearRegression.on_iris()
