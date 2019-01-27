import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron


def load_data():
    """
    加载iris模型，只选择其中两类作为训练集
    :return:
    """
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['label'] = iris.target
    iris_df = iris_df[(iris_df['label'] != 2)]
    iris_df['label'][(iris_df['label'] == 0)] = -1
    return iris_df


def plot_perceptron(data, model):
    """
    绘制感知机模型
    :param data: 训练数据. [dataframe]
    :param model: 训练好的模型. [model]
    :return:
    """
    plt.plot(data.iloc[:, 0][(data['label'] == -1)], data.iloc[:, 1][(data['label'] == -1)], 'bo', color='blue',
             label='0')
    plt.plot(data.iloc[:, 0][(data['label'] == 1)], data.iloc[:, 1][(data['label'] == 1)], 'bo', color='orange',
             label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()

    x_ponits = np.arange(4, 8)
    y_ = -(model.coef_[0][0] * x_ponits + model.intercept_) / model.coef_[0][1]
    plt.plot(x_ponits, y_)
    plt.show()


def main():
    iris_df = load_data()
    clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
    clf.fit(iris_df.iloc[:, [0, 1]], iris_df.iloc[:, -1])
    plot_perceptron(iris_df, clf)


if __name__ == '__main__':
    main()
