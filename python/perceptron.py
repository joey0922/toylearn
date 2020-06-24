#! /usr/env/bin python
# -*- encoding:utf-8 -*-

import numpy as np
from copy import deepcopy

# simple implement of perceptron algorithm
class Perceptron:

    def __init__(self, learning_rate=1.0, random_state=0, dual_form=False,
                 batch_size=1, n_iter_no_change=5, max_iter=1000, shuffle=True):

        if not isinstance(learning_rate, (int, float)) or not 0 < learning_rate <= 1:
            raise ValueError('Learning rate is numeric greater than 0 '
                             'and not greater than 1.')
        self.__learning_rate = learning_rate

        if not isinstance(random_state, (int, float)):
            raise ValueError(f"Parameter 'init_state' must be numeric number, "
                             f"not {type(random_state)}.")
        self.__random_state = random_state

        if not isinstance(dual_form, bool):
            raise ValueError("Type of parameter 'dual_form' must be bool.")
        self.__dual_form = dual_form

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Batch size must be an int number and greater than 0.')
        self.__batch_size = batch_size

        if not isinstance(n_iter_no_change, int) or n_iter_no_change <= 1:
            raise ValueError("'n_iter_no_change' must an int number and greater than 1.")
        self.__n_iter_no_change = n_iter_no_change

        if not isinstance(max_iter, int) or max_iter <= 1:
            raise ValueError('Max iteration must be an int number and greater than 1.')
        self.__max_iter = max_iter

        if not isinstance(shuffle, bool):
            raise ValueError(f'Type of parameter "shuffle" should be bool, not {type(shuffle)}.')
        self.__shuffle = shuffle

        self.__w = None
        self.__b = None

    @property
    def coef_(self):
        return self.__w

    @property
    def bias(self):
        return self.__b

    @property
    def learning_rate(self):
        return self.__learning_rate

    @property
    def random_state(self):
        return self.__random_state

    @property
    def dual_form(self):
        return self.__dual_form

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def n_iter_no_change(self):
        return self.__n_iter_no_change

    @property
    def max_iter(self):
        return self.__max_iter

    @property
    def shuffle(self):
        return self.__shuffle

    @learning_rate.setter
    def learning_rate(self, lr):
        if not isinstance(lr, float) or not 0 < lr <= 1:
            raise ValueError('Learning rate is a float number greater than 0 '
                             'and not greater than 1.')
        self.__learning_rate = lr

    @random_state.setter
    def random_state(self, rs):
        if not isinstance(rs, (int, float)):
            raise ValueError(f"Parameter 'init_state' must be numeric number, "
                             f"not {type(rs)}.")
        self.__random_state = rs

    @dual_form.setter
    def dual_form(self, df):
        if not isinstance(df, bool):
            raise ValueError("Type of parameter 'df' user input is not bool. ")
        self.__dual_form = df

    @batch_size.setter
    def batch_size(self, bs):
        if not isinstance(bs, int) or bs < 1:
            raise ValueError('Batch size must be an int number and greater than 0.')
        self.__batch_size = bs

    @n_iter_no_change.setter
    def n_iter_no_change(self, ninc):
        if not isinstance(ninc, int) and ninc <= 0:
            raise ValueError("'n_iter_no_change' must an int number and greater than 1.")
        self.__n_iter_no_change = ninc

    @max_iter.setter
    def max_iter(self, mi):
        if not isinstance(mi, int) or mi <= 1:
            raise ValueError('Max iteration must be an int number and greater than 1.')
        self.__max_iter = mi

    @shuffle.setter
    def shuffle(self, sf):
        if not isinstance(sf, bool):
            raise ValueError(f'Type of parameter "shuffle" should be bool, not {type(sf)}.')
        self.__shuffle = sf

    def fit(self, train_set, labels):

        # parameter check
        if not isinstance(train_set, np.ndarray):
            train_set = np.array(train_set)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        if train_set.shape[0] != labels.shape[0] or len(train_set.shape) != 2 \
                or len(labels.shape) != 1:
            raise ValueError('Shape of train_set or labels is wrong or not equal.')

        # train
        if self.__dual_form:
            self.__w, self.__b = self.__dual_fit(train_set, labels)
        else:
            self.__w, self.__b = self.__origin_fit(train_set, labels)

        return self

    # 原始形式
    def __origin_fit(self, train_set, labels):

        w, b = self.__random_state, self.__random_state
        tmp_w, tmp_b = self.__random_state, self.__random_state
        delta_w, delta_b = 0, 0

        idx_list = np.array([i for i in range(labels.shape[0])])
        if self.__shuffle:
            np.random.shuffle(idx_list)
        length = idx_list.shape[0]

        loop_index, no_change_turn = 0, 0
        i, err_batch = 0, 0
        while True:
            loop_index += 1

            idx = idx_list[i]
            if labels[idx] * ((w * train_set[idx]).sum() + b) <= 0:
                delta_w += self.__learning_rate * labels[idx] * train_set[idx]
                delta_b += self.__learning_rate * labels[idx]
                err_batch += 1
            if err_batch == self.__batch_size:
                w += delta_w / self.__batch_size
                b += delta_b / self.__batch_size
                delta_w, delta_b, err_batch = 0, 0, 0

            if (tmp_w == w).all() and tmp_b == b:
                no_change_turn += 1
            else:
                tmp_w, tmp_b = deepcopy(w), b
                if no_change_turn != 0:
                    no_change_turn = 0
            if no_change_turn == self.__n_iter_no_change:
                break

            if loop_index == self.__max_iter:
                break

            i += 1
            if i == length:
                i = 0
                if self.__shuffle:
                    np.random.shuffle(idx_list)

        return w, b

    def __dual_fit(self, train_set, labels):

        gram_matrix = np.dot(train_set, train_set.T)

        alpha = np.array([self.__random_state]*labels.shape[0]).astype('float64')
        w = sum((alpha * labels).reshape(-1, 1) * train_set)
        b = self.__random_state

        tmp_w, tmp_b = deepcopy(w), self.__random_state
        delta_alpha, delta_b = 0, 0

        idx_list = np.array([i for i in range(labels.shape[0])])
        if self.__shuffle:
            np.random.shuffle(idx_list)
        length = idx_list.shape[0]

        loop_index, no_change_turn = 0, 0
        i, err_batch = 0, 0
        while True:
            loop_index += 1
            print(loop_index, w, b)

            idx = idx_list[i]
            if labels[idx] * (np.dot(alpha*labels, gram_matrix[:, idx]) + b) <= 0:
                delta_alpha += self.__learning_rate
                delta_b += self.__learning_rate * labels[idx]
                err_batch += 1
            if err_batch == self.__batch_size:
                alpha += delta_alpha / self.__batch_size
                b += delta_b / self.__batch_size
                delta_alpha, delta_b, err_batch = 0, 0, 0

            w = sum((alpha * labels).reshape(-1, 1) * train_set)
            if (tmp_w == w).all() and tmp_b == b:
                no_change_turn += 1
            else:
                tmp_w, tmp_b = deepcopy(w), b
                if no_change_turn != 0:
                    no_change_turn = 0
            if no_change_turn == self.__n_iter_no_change:
                break

            if loop_index == self.__max_iter:
                break

            i += 1
            if i == length:
                i = 0
                if self.__shuffle:
                    np.random.shuffle(idx_list)

        return w, b

    def predict(self, x):

        if self.__w is None or self.__b is None:
            raise ValueError('Parameter w or b is None.')

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if len(x.shape) == 1:
            return 1 if (self.__w*x).sum()+self.__b > 0 else -1
        elif len(x.shape) == 2:
            labelist = (self.__w*x).sum(axis=1) + self.__b
            return np.array([1 if i > 0 else -1 for i in labelist])
        else:
            raise ValueError("Shape of user input 'x' must be (n, m) or (n, ).")

    def fit_predict(self, x, train_set, labels):
        self.fit(train_set, labels)
        label = self.predict(x)
        return label


if __name__ == '__main__':

    data_set = np.array([[3, 3], [4, 3], [1, 1]])
    label_set = np.array([1, 1, -1])
    tron = Perceptron(dual_form=True, max_iter=1000)
    model = tron.fit(data_set, label_set)
    print(model.coef_, model.bias)
    # gram = np.dot(data_set, data_set.T)
    # print(1 * label_set*gram[0, :])
    # print(np.dot(1 * label_set, gram[0, :]))