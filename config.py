from easydict import EasyDict

# ----公共参数----
model_select = 'naive_bayes'  # 选取的模型，可选'perceptron'、'knn'、'naive_bayes'
save_model = True  # 是否保存模型
save_path = './saves'  # 模型保存路径
test_size = 0.1   # 测试集占比

# ----感知机模型参数----
perceptron_config = EasyDict()
perceptron_config.fit_intercept = False   # 是否拟合常数项
perceptron_config.max_iter = 1000   # 最大迭代次数
perceptron_config.shuffle = False   # 是否对数据进行打散

# ----KNN模型参数----
knn_config = EasyDict()
knn_config.n_neighbors=5   # K值大小
knn_config.algorithm='auto'   # 实现算法，可选'auto', 'ball_tree', 'kd_tree', 'brute'
knn_config.p=2   # Lp距离公式中的p值，当p=2表示欧式距离

# ----朴素贝叶斯模型参数----
bayes_config = EasyDict()
bayes_config.classfier = 'gaussian'   # 当变量是离散型时，选择'multinomial',当特征变量是连续型时，则选择'gaussian'
bayes_config.alpha = 1.0   # 平滑项，只有当classfier选择multinomial时才需要用到


