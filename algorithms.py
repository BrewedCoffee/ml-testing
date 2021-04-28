from scipy import stats
from matplotlib import pyplot
import numpy
import pandas

import sklearn.preprocessing
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import sklearn

import data

class Algorithm:
    # data = data.Data()

    @classmethod
    def on_iris(cls):
        raise NotImplementedError
    
    @classmethod
    def on_data(cls, data, **kwargs):
        raise NotImplementedError

class Classifier(Algorithm):
    @staticmethod
    def encode_labels(data, col):
        category_encoder = sklearn.preprocessing.LabelEncoder()
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
    def test(cls):
        raise NotImplementedError

class LinearRegression(Algorithm):
    @classmethod
    def on_iris(cls):
        return cls.fit(
            cls.data.iris['Sepal Length'].to_list(),
            cls.data.iris['Sepal Width'].to_list())

    @staticmethod
    def on_data(data, columns):
        return LinearRegression.fit_n(data.encoded, columns)

    @staticmethod
    def fit_n(data, columns):
        pass

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
    def on_data(data: data.ClassificationData, multi_class='auto', solver='lbfgs'):
        train_data, test_data = sklearn.model_selection.train_test_split(data.encoded)
        model = LogisticRegression.train(data.get_independent(train_data), data.get_dependent(train_data), multi_class, solver)
        print(LogisticRegression.test(model, data.get_independent(test_data), data.get_dependent(test_data)))

    @staticmethod
    def train(train_x, train_y, multi_class, solver):
        model = sklearn.linear_model.LogisticRegression(multi_class=multi_class, solver=solver)
        return model.fit(train_x, train_y)

    @staticmethod
    def test(model, test_x, test_y):
        return f'{model.score(test_x, test_y)}\n{sklearn.metrics.confusion_matrix(test_y, model.predict(test_x))}\n{sklearn.metrics.classification_report(test_y, model.predict(test_x))}'


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

def main():
    # LogisticRegression.on_iris()
    LogisticRegression.on_data(data.IrisData())
    # LinearRegression.on_data(data.)
    # LinearRegression.on_iris()

if __name__ == '__main__':
    main()
