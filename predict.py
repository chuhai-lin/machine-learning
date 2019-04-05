import os
import config
from data_loader import DataLoader
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def predict(x,
            model_select=config.model_select,
            save_path=config.save_path
            ):
    # 加载模型
    model_path = os.path.join(save_path, model_select, 'model.model')
    model = joblib.load(model_path)

    # 预测
    pred = model.predict(x)
    return pred


if __name__ == '__main__':
    # 加载数据集
    data_loader = DataLoader()
    data = data_loader.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                                                        test_size=config.test_size, random_state=1234)

    # 训练模型
    pred = predict(x=x_test)

    # 计算预测准确率
    acc = accuracy_score(y_test, pred)
    print('预测准确率：{0}'.format(acc))
