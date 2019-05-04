from easydict import EasyDict

# ----公共参数----
model_select = 'max_entropy'  # 选取的模型，可选'perceptron'、'knn'、'naive_bayes','cart','logistic','max_entropy'
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

# ----CART决策树模型----
cart_config = EasyDict()
cart_config.max_depth = 5   # 树的最大深度
cart_config.min_samples_leaf = 5   # 每个叶结点必须包括的最小的样本数量
cart_config.draw_tree = True   # 适合绘制决策树

# ----Logistic回归模型----
lr_config = EasyDict()
lr_config.max_iter = 200   # 最大的迭代次数

# ----Max_Entropy模型----
max_entropy_config = EasyDict()
max_entropy_config.eps = 0.005   # 参数收敛阈值
max_entropy_config.maxiter = 1000   # 最大迭代次数

