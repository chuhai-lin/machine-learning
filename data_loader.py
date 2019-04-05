import pandas as pd
from sklearn.datasets import load_iris

class DataLoader(object):
    def __init__(self):
        pass

    def load_iris(self):
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