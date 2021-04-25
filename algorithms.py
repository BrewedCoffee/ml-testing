from scipy import stats
from matplotlib import pyplot
import matplotlib.pyplot as pyplot
import numpy
import pandas
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

import data

class Algorithm:
    data = data.Data()

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
    def get_label_encoder(categories):
        category_encoder = preprocessing.LabelEncoder()
        category_encoder.fit(categories)
        return category_encoder

    @classmethod
    def on_iris(cls):
        iris_classes = (data.IRIS_CLASSES[0], data.IRIS_CLASSES[1])
        iris_dataframe = cls.data.iris.loc[cls.data.iris['Class'].isin(iris_classes)]
        iris_data_array = iris_dataframe.to_numpy()
        label_encoder = LogisticRegression.get_label_encoder(numpy.asarray(iris_classes))
        iris_data_array[:, -1] = label_encoder.transform(iris_data_array[:,-1])

    @staticmethod
    def train(x, y):
        print(x, y)

    @staticmethod
    def classify(model, x):
        pass

LogisticRegression.on_iris()
