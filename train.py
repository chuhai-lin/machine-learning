import os
import config
from data_loader import DataLoader
from sklearn.externals import joblib
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.model_selection import train_test_split
from config import perceptron_config,knn_config,bayes_config,cart_config,lr_config
from models.tree import DecisionTreeClassifier


def train(x, y,
          model_select=config.model_select,
          save_model=config.save_model,
          save_path=config.save_path
          ):
    # 感知机模型
    if model_select == 'perceptron':
        model = Perceptron(fit_intercept=perceptron_config.fit_intercept,
                           max_iter=perceptron_config.max_iter,
                           shuffle=perceptron_config.shuffle)
    # KNN模型
    elif model_select == 'knn':
        model = KNeighborsClassifier(n_neighbors=knn_config.n_neighbors,
                                     algorithm=knn_config.algorithm,
                                     p=knn_config.p)
    # 朴素贝叶斯模型
    elif model_select == 'naive_bayes':
        if bayes_config.classfier == 'gaussian':
            model = GaussianNB()
        elif bayes_config.classfier == 'multinomial':
            model = MultinomialNB(alpha=bayes_config.alpha)
    # CART决策树
    elif model_select == 'cart':
        model = DecisionTreeClassifier(max_depth=cart_config.max_depth,
                                       min_samples_leaf=cart_config.min_samples_leaf)
    # Logistic回归模型
    elif model_select == 'logistic':
        model = LogisticRegression(max_iter=lr_config.max_iter)

    # 训练模型
    model.fit(data.iloc[:, :-1], data.iloc[:, -1])

    # 保存模型
    if save_model:
        model_path = os.path.join(save_path, model_select)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(model, os.path.join(model_path, 'model.model'))

    # 是否绘制决策树，只有选择CART时才有效
    if model_select == 'cart' and cart_config.draw_tree:
        save_path = os.path.join(save_path, model_select,'tree.dot')
        model.draw_tree(model, feature_names=data.iloc[:,:-1].columns.tolist(), save_path=save_path)


if __name__ == '__main__':
    # 加载数据集
    data_loader = DataLoader()
    data = data_loader.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                                                        test_size=config.test_size, random_state=1234)

    # 训练模型
    train(x_train, y_train)
