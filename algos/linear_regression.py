import scipy.stats as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import Data

def process_data(df):
    x = df['Sepal Length'].to_list()
    y = df['Sepal Width'].to_list()

    result = sp.linregress(x, y)

    print(f"R-squared: {result.rvalue**2:.6f}")

    for i in range(len(y)):
        y[i] = result.intercept + result.slope*x[i]

    plt.plot(x, y, 'o', label='original data')
    plt.plot(x, y, 'r', label='fitted line')
    plt.show()

data_manager = Data()
process_data(data_manager.iris)