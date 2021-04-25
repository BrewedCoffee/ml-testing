from scipy import stats
from matplotlib import pyplot
import numpy
import pandas
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
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

class NaiveBayes(Classifier):
    @classmethod
    def test(self):
        array = self.data.iris
        X = array.iloc[:,1:5]
        Y = array[:,5]

        X_train, X_validation, Y_train, Y_validation = sklearn.model_selection.train_test_split(
            X, Y, test_size=0.2, random_state=7
        )

        model = sklearn.naive_bayes.GaussianNB()
        model.fit(X_train, Y_train)

        for i in range(len(X_validation)):
            print(model.predict(X_validation[i]))
        # prob_predict = gnb.predict_proba(X_validation)[:, 1]
        # sklearn.naive_bayes

# NaiveBayes.test()
LogisticRegression.on_iris()

LinearRegression.on_iris()
