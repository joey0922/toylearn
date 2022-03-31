#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import joblib
import numpy as np


'''
优点：泛化错误率低，计算开销不大，结果易解释。
缺点：对参数调节和核函数的选择敏感，原始分类器不加修改仅适用于处理二类问题。
'''


# 核函数类
class Kernel(object):

    def __init__(self):
        pass

    # 线性核函数
    def linear(self, x, y=None):
        '''
        :param x: feature array
        :param y: feature array
        :return: K(x, y) = x.T * y
        '''
        if y is None:
            y = x

        return (np.mat(x) * np.mat(y).T).A

    # 径向基核函数
    def rbf(self, x, y=None, gamma=None):
        '''
        :param x: feature array, (n_examples, n_features)
        :param y: feature array, (n_examples, n_features)
        :param gamma: kernel coefficient, if None, uses 1 / n_features
        :return: K(x, y) = exp(-gamma ||x-y||^2)
        '''
        x = np.mat(x)
        y = np.mat(y) if y is not None else x

        if gamma is None:
            gamma = 1 / x.shape[1]

        return np.array([[np.exp(-gamma * np.square(xi - yi).sum())
                          for yi in y] for xi in x])

    # 多项式核函数
    def poly(self, x, y=None, degree=3, gamma=None, coef0=1):
        '''
        :param x: feature array, (n_examples, n_features)
        :param y: feature array, (n_examples, n_features)
        :param degree: degree of the polynomial kernel function, int, default=3
        :param gamma: kernel coefficient, float, default=None
        :param coef0: independent term in kernel function, float, default=1
        :return: K(x, y) = (gamma * x.T * y + coef0) ^ degree
        '''
        if y is None:
            y = x

        if gamma is None:
            gamma = 1 / x.shape[1]

        return np.power(gamma * np.mat(x) * np.mat(y).T + coef0, degree).A

    # sigmoid核函数
    def sigmoid(self, x, y=None, gamma=None, coef0=1):
        '''

        :param x: feature array, (n_examples, n_features)
        :param y: feature array, (n_examples, n_features)
        :param gamma: kernel coefficient, float, default=None
        :param coef0: independent term in kernel function, float, default=1
        :return: K(x, y) = tanh(gamma * x.T * y + coef0)
        '''
        if y is None:
            y = x

        if gamma is None:
            gamma = 1 / x.shape[-1]

        return np.tanh(gamma * np.mat(x) * np.mat(y).T + coef0).A

    # 拉普拉斯核函数
    def laplacian(self, x, y=None, gamma=None):
        '''
        :param x: feature array, (n_examples, n_features)
        :param y: feature array, (n_examples, n_features)
        :param gamma: kernel coefficient, float, default=None
        :return: K(x, y) = exp(-gamma * ||x - y||_1)
        '''
        x = np.mat(x)
        y = np.mat(y) if y is not None else x

        if gamma is None:
            gamma = 1 / x.shape[-1]

        return np.array([[np.exp(-gamma * np.abs(xi - yi).sum())
                          for yi in y] for xi in x])

    # 卡方核函数
    def chi2(self, x, y=None, gamma=1):
        '''
        :param x: feature array, (n_examples, n_features)
        :param y: feature array, (n_examples, n_features)
        :param gamma: kernel coefficient, float, default=1
        :return: K(x, y) = exp(-gamma * sum [(x - y)^2 / (x + y)])
        '''
        x = np.mat(x)
        y = np.mat(y) if y is not None else x

        return np.array([[np.exp(-gamma * (np.square(xi - yi) / (xi + yi)).sum())
                          for yi in y] for xi in x])

    # 余弦相似度
    def cosine_similarity(self, x, y=None):
        '''
        :param x: feature array, (n_examples, n_features)
        :param y: feature array, (n_examples, n_features)
        :return: K(x,y) = (x * y.T) / (||x|| * ||y||)
        '''
        x = np.mat(x)
        y = np.mat(y) if y is not None else x

        xm = np.sqrt(np.square(x).sum(1)).A.flatten()
        ym = np.sqrt(np.square(y).sum(1)).A.flatten()
        xym = np.array([[xi*yi for yi in ym] for xi in xm])

        return (x * y.T / xym).A


# 数据存储类
class OptData:

    def __init__(self, data_mat, labels, kernel_name, *,
                 gamma=None, coef0=None, degree=None):

        if type(data_mat) is not np.matrix:
            data_mat = np.mat(data_mat)
        if type(labels) is not np.matrix:
            labels = np.mat(labels)
        if labels.shape[1] != 1:
            labels = labels.transpose()

        self.data_mat = data_mat
        self.labels = labels
        self.n_rows = data_mat.shape[0]
        self.error_cache = np.mat(np.zeros((self.n_rows, 2)))

        self.kernel_name = kernel_name
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_cache = np.mat(calc_kernel_matrix(self))


# 计算核函数矩阵函数
def calc_kernel_matrix(opt):

    params_dict = {'rbf': ('gamma',), 'poly': ('gamma', 'coef0', 'degree'),
                   'sigmoid': ('gamma', 'coef0'), 'laplacian': ('gamma',),
                   'chi2': ('gamma',)}

    params = params_dict.get(opt.kernel_name, None)
    if params is not None:
        params = {key: getattr(opt, key, None) for key in params if getattr(opt, key, None)}

    if params:
        return getattr(Kernel(), opt.kernel_name)(opt.data_mat, **params)
    else:
        return getattr(Kernel(), opt.kernel_name)(opt.data_mat)


# 计算误差缓存E
def calc_Ek(k, model, opt):
    '''
    :param k: index of data
    :param model: svm model
    :param opt: OptData model
    :return: error cache
    E = sum([alpha_i * y_i * K(x_i, x_k) for i in range(len(data)]) + b - y_k
    '''

    fxk = float(model.alphas.T.A * opt.labels.T.A * opt.kernel_cache[:, k]) + \
          model.bias
    Ek = fxk - float(opt.labels[k])

    return Ek


# 更新误差缓存E
def update_Ek(k, model, opt):
    Ek = calc_Ek(k, model, opt)
    opt.error_cache[k, 0], opt.error_cache[k, 1] = 1, Ek


# 数值截断，使值在[low, high]范围内
def clip(value, low, high):

    if value > high:
        value = high
    if value < low:
        value = low

    return value


# 随机选择内循环索引
def rand_select_index(i, high):

    j = i
    while j == i:
        j = np.random.randint(0, high)

    return j


# 选择内循环索引
def select_index(i, Ei, model, opt):
    opt.error_cache[i, 0],  opt.error_cache[i, 1] = 1, Ei

    max_j, max_delta_ej, Ej = -1, 0, 0
    valid_cache_list = np.nonzero(opt.error_cache[:, 0].A)[0]
    if len(valid_cache_list) > 1:
        for k in valid_cache_list:
            if k == i: continue
            Ek = calc_Ek(k, model, opt)
            delta_ek = np.abs(Ei - Ek)
            if delta_ek > max_delta_ej:
                max_j = k
                max_delta_ej = delta_ek
                Ej = Ek
    else:
        max_j = rand_select_index(i, opt.n_rows)
        Ej = calc_Ek(max_j, model, opt)

    return max_j, Ej


# 内层循环
def inner_loop(i, model, opt):
    Ei = calc_Ek(i, model, opt)
    # 违反KKT条件判断
    if (opt.labels[i] * Ei < -model.tol and model.alphas[i] < model.C) \
            or (opt.labels[i] * Ei > model.tol and model.alphas[i] > 0):

        j, Ej = select_index(i, Ei, model, opt)
        alpha_i_old, alpha_j_old = model.alphas[i].copy(), model.alphas[j].copy()

        # labels[i]==labels[j]时，alpha_i与alpha_j的和为常数；否则，alpha_i与alpha_j差为常数。
        if opt.labels[i] != opt.labels[j]:
            L = max(0, model.alphas[j] - model.alphas[i])
            H = min(model.C, model.C + model.alphas[j] - model.alphas[i])
        else:
            L = max(0, model.alphas[j] + model.alphas[i] - model.C)
            H = min(model.C,  model.alphas[j] + model.alphas[i])
        if L == H:
            # print("L == H")
            return 0

        eta = 2.0 * opt.kernel_cache[i, j] - opt.kernel_cache[i, i] - opt.kernel_cache[j, j]
        if eta >= 0:
            # print("eta >= 0")
            return 0

        # 更新alpha j
        model.alphas[j] -= opt.labels[j] * (Ei - Ej) / eta
        model.alphas[j] = clip(model.alphas[j], L, H)
        update_Ek(j, model, opt)
        if np.abs(model.alphas[j], alpha_j_old) < 1e-5:
            # print('Inner alpha not moving enough.')
            return 0

        # 更新alpha i
        model.alphas[i] += opt.labels[i] * opt.labels[j] * (alpha_j_old - model.alphas[j])
        update_Ek(i, model, opt)

        # 更新bias
        b1 = model.bias - Ei - opt.labels[i] * (model.alphas[i] - alpha_i_old) * \
            opt.kernel_cache[i, i] - opt.labels[j] * (model.alphas[j] - alpha_j_old) *\
            opt.kernel_cache[i, j]
        b2 = model.bias - Ej - opt.labels[i] * (model.alphas[i] - alpha_i_old) * \
            opt.kernel_cache[i, j] - opt.labels[j] * (model.alphas[j] - alpha_j_old) *\
            opt.kernel_cache[j, j].T
        if 0 < model.alphas[i] < model.C:
            model.bias = b1
        elif 0 < model.alphas[j] < model.C:
            model.bias = b2
        else:
            model.bias = (b1 + b2) / 2

        return 1
    else:
        return 0


# 序列最小优化算法(sequential minimal optimization)
def SMO(model, opt):

    model.alphas = np.mat(np.zeros((opt.n_rows, 1)))
    model.bias = 0

    loop = 0
    entire_set = True
    alpha_pairs_changed = 0
    while loop < model.max_iter and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(opt.n_rows):
                alpha_pairs_changed += inner_loop(i, model, opt)
                # print(f'Full set, iter: {loop}, i: {i}, '
                #       f'{alpha_pairs_changed} pairs changed')
        else:
            non_bound_is = np.nonzero((model.alphas > 0).A *
                                      (model.alphas < model.C).A)[0]
            for i in non_bound_is:
                alpha_pairs_changed += inner_loop(i, model, opt)
                # print(f'non-bound, iter: {loop}, i: {i}, '
                #       f'{alpha_pairs_changed} pairs changed')

        loop += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True

        # print(f'Iteration number: {loop}')


# 支持向量机
class SVC(object):

    def __init__(self, C=1.0, kernel='rbf', max_iter=-1, tol=1e-3,
                 gamma=None, coef0=1, degree=3):

        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

        self.kernel = kernel
        self.alphas = None
        self.bias = 0

        self.n_support_ = 0
        self.support_vectors = None
        self.support_labels = None
        self.weights = None

    def fit(self, inps, labels):

        opt = OptData(inps, labels, self.kernel,
                      gamma=self.gamma, coef0=self.coef0, degree=self.degree)
        SMO(self, opt)

        sup_idx = np.nonzero(self.alphas)[0]
        self.support_vectors = inps[sup_idx]
        self.n_support_ = self.support_vectors.shape[0]
        self.support_labels = labels[sup_idx]
        self.alphas = self.alphas[sup_idx]
        self.weights = np.mat(self.support_labels * self.alphas.T.A)
        self.bias = float(self.bias)

    def predict(self, x, proba=False):

        params_dict = {'rbf': ('gamma',), 'poly': ('gamma', 'coef0', 'degree'),
                       'sigmoid': ('gamma', 'coef0'), 'laplacian': ('gamma',),
                       'chi2': ('gamma',)}

        params = params_dict.get(self.kernel, None)
        if params is not None:
            params = {key: getattr(self, key, None)
                      for key in params if getattr(self, key, None)}

        if params:
            k_arr = getattr(Kernel(), self.kernel)(self.support_vectors, x, **params)
        else:
            k_arr = getattr(Kernel(), self.kernel)(self.support_vectors, x)

        sign = (self.weights * np.mat(k_arr) + self.bias).A[0]
        
        return sign / np.abs(sign)

    def save(self, path):
        return joblib.dump(self, path)


if __name__ == '__main__':

    from sklearn.datasets import load_iris

    def get_bi_iris():
        iris = load_iris()
        data = [(iris.data[i], l) for i, l in enumerate(iris.target) if l == 1 or l == 0]
        np.random.shuffle(data)
        X, y = list(zip(*data))

        return np.array(X), np.array(y)


    x, y = get_bi_iris()
    y[y == 0] = -1

    model = SVC(max_iter=10000)
    model.fit(x, y)
    print(model.predict(x))
    print(all(model.predict(x)==y))