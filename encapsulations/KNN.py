import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

class KNNClassifier:

    def __init__(self,k):
        """初始化KNN分类器"""
        assert k >= 1,"k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self,X_train,y_train):
        """根据训练集X_train和y_train训练KNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must equal to the size of y_train"
        assert self.k <= X_train.shape[0], "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self,X_predict):
        """给定待预测数据集X_predict,返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._X_train is not None,\
            "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1],\
            "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self,x):
        """给单个的待预测数据x,返回x_predict的预测结果值"""
        assert x.shape[0] == self._X_train.shape[1],\
            "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x - x_train)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)[:self.k]
        topK_y = self._y_train[nearest]
        votes = Counter(topK_y).most_common()[0][0]
        return votes


    def score(self,X_test,y_test):
        """根据测试数据集 X_test和y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test,y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k













