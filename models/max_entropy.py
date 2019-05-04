import math
import numpy as np


class MaxEntropy(object):
    def __init__(self, eps=0.005, maxiter=1000):
        self._Y = set()  # 标签集合，相当去去重后的y
        self._numXY = {}  # key为(x,y)，value为出现次数
        self._N = 0  # 样本数
        self._Ep_ = []  # 样本分布的特征期望值
        self._xyID = {}  # key记录(x,y),value记录id号
        self._n = 0  # 特征键值(x,y)的个数
        self._C = 0  # 最大特征数
        self._IDxy = {}  # key为id号，value为对应的(x,y)
        self._w = []
        self._eps = eps  # 收敛条件
        self._lastw = []  # 上一次w参数值
        self.maxiter = maxiter

    def _Zx(self, X):
        """计算每个Z(x)值"""
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            zx += math.exp(ss)
        return zx

    def _model_pyx(self, y, X):
        """计算每个P(y|x)"""
        zx = self._Zx(X)
        ss = 0
        for x in X:
            if (x, y) in self._numXY:
                ss += self._w[self._xyID[(x, y)]]
        pyx = math.exp(ss) / zx
        return pyx

    def _model_ep(self, X, index):
        """计算特征函数fi关于模型的期望"""
        x, y = self._IDxy[index]
        ep = 0
        for sample in X:
            if x not in sample:
                continue
            pyx = self._model_pyx(y, sample)
            ep += pyx / self._N
        return ep

    def _convergence(self):
        """判断是否全部收敛"""
        for last, now in zip(self._lastw, self._w):
            if abs(last - now) >= self._eps:
                return False
        return True

    def predict(self, X):
        """计算预测概率"""
        result = []
        X = np.asarray(X).tolist()
        for x in X:
            Z = self._Zx(x)
            logit = {}
            for y in self._Y:
                ss = 0
                for xi in x:
                    if (xi, y) in self._numXY:
                        ss += self._w[self._xyID[(xi, y)]]
                pyx = math.exp(ss) / Z
                logit[y] = pyx
            logit = sorted(logit.items(), key=lambda x: x[1], reverse=True)
            result.append(logit[0][0])
        return result

    def fit(self, X, Y):
        """训练模型"""
        X = np.asarray(X).tolist()
        Y = np.asarray(Y).tolist()
        for x, y in zip(X, Y):
            # 集合中y若已存在则会自动忽略
            self._Y.add(y)
            for xi in x:
                if (xi, y) in self._numXY:
                    self._numXY[(xi, y)] += 1
                else:
                    self._numXY[(xi, y)] = 1

        self._N = len(X)
        self._n = len(self._numXY)
        self._C = max([len(sample) - 1 for sample in X])
        self._w = [0] * self._n
        self._lastw = self._w[:]

        # 计算特征函数fi关于经验分布的期望
        self._Ep_ = [0] * self._n
        for i, xy in enumerate(self._numXY):
            self._Ep_[i] = self._numXY[xy] / self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

        # 更新模型的参数
        for loop in range(self.maxiter):
            print("iter:%d" % loop)
            self._lastw = self._w[:]
            for i in range(self._n):
                ep = self._model_ep(X, i)  # 计算第i个特征的模型期望
                self._w[i] += math.log(self._Ep_[i] / ep) / self._C  # 更新参数
            print("w:", self._w)
            if self._convergence():  # 判断是否收敛
                break