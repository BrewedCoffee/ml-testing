import scipy.stats as stats
import matplotlib.pyplot as pyplot
import numpy
import pandas

from data import Data

data = Data()

class LinearRegression:
    def fit(df):
        x = df['Sepal Length'].to_list()
        y = df['Sepal Width'].to_list()

        result = stats.linregress(x, y)

        print(f"R-squared: {result.rvalue**2:.6f}")
        pyplot.plot(x, y, 'o', label='original data')

        for i in range(len(y)):
            y[i] = result.intercept + result.slope*x[i]

        pyplot.plot(x, y, 'r', label='fitted line')
        pyplot.show()

class LogisticRegression:
    def train(data):
        pass

    def classify(model, data):
        pass
