#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import joblib
import numpy as np


'''
优点：计算代价不高，易于理解和实现。
缺点：容易欠拟合，分类精度可能不高。
'''


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 逻辑回归
class LogisticRegression(object):

    def __init__(self, weights=None, bias=0, alpha=1e-3, rand_init=True,
                 penalty='l2', pen_coef=0.5, max_iter=100, tol=1e-4):

        self.weights = weights
        self.bias = bias
        self.alpha = alpha
        self.penalty = penalty.lower() if isinstance(penalty, str) else penalty
        self.pen_coef = pen_coef
        self.max_iter = max_iter
        self.tol = tol
        self.rand_init = rand_init

    def fit(self, inps, labels):
        inps = np.mat(inps)
        labels = np.mat(labels).transpose()

        n_row, n_col = inps.shape
        if self.weights is None:
            if self.rand_init:
                self.weights = np.mat(np.random.rand(n_col)).T
            else:
                self.weights = np.mat(np.ones(n_col)).T

        if self.bias is not None:
            inps = np.concatenate((inps, np.mat(np.ones(n_row)).T), axis=1)
            self.weights = np.concatenate((self.weights, np.mat([[self.bias]])))

        if self.penalty == 'l1':
            l1_dev = {True: 1, False: -1}
            
        for _ in range(1, self.max_iter+1):
            y_hat = sigmoid(inps * self.weights)
            loss = inps.T * (y_hat - labels) / n_row
            if self.penalty == 'l2':
                loss += self.pen_coef * self.weights / n_row
            if self.penalty == 'l1':
                dev = [e if e == 0 else l1_dev[e > 0] for e in self.weights.T[0]][0].T
                loss += self.pen_coef * dev / n_row
            loss *= self.alpha
            self.weights -= loss
        
        if self.bias is not None:
            self.bias = self.weights[-1].getA()[0,0]
            self.weights = self.weights[:-1]

    def predict(self, x, proba=True):

        x = np.mat(x)

        proba_list = sigmoid(x * self.weights + self.bias)
        y_hat = (proba_list > 0.5).astype(int)

        if proba:
            y_hat = np.concatenate((y_hat, proba_list), 1).getA()
        else:
            y_hat = y_hat.getA().T[0]

        return y_hat

    def save(self, path):
        if self.weights is None:
            raise ValueError('Weights of model is None, train to get them first!')

        return joblib.dump(self, path)

    def load(self, path):
        model = joblib.load(path)
        self.weights = model.weights
        self.bias = model.bias

    def save_params_dict(self, path):
        if self.weights is None:
            raise ValueError('Weights of model is None, train to get them first!')

        return joblib.dump({'weights': self.weights, 'bias': self.bias}, path)

    def load_params_dict(self, path):
        params_dict = joblib.load(path)
        self.weights = params_dict.get('weights', None)
        self.bias = params_dict.get('bias', None)

if __name__ == '__main__':

    from sklearn.datasets import load_iris

    def get_bi_iris():
        iris = load_iris()
        data = [(iris.data[i], l) for i, l in enumerate(iris.target) if l == 1 or l == 0]
        np.random.shuffle(data)
        x, y = list(zip(*data))

        return np.array(x), np.array(y)

    x, y = get_bi_iris()

    lr = LogisticRegression(max_iter=1000)
    lr.fit(x, y)
    y_hat = lr.predict(x, proba=False)
    print(all(y_hat == y))