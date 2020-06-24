#! /usr/env/bin python
# -*- encoding:utf-8 -*-

import numpy as np
import warnings

'''
优点：精度高、对异常值不敏感、无数据输入假定
缺点：计算复杂度高、空间复杂度高
'''


# simple implement of KNN algorithm
class KNN:

    def __init__(self, k=3, data_set=None, labels=None):

        if not isinstance(k, int):
            if isinstance(k, float):
                warnings.warn('Type of parameter "k" is float, will transfer to an int.')
                k = int(k)
            else:
                raise ValueError(f'Type of parameter "k" must be numeric instead of {type(k)}.')
        self.__k = k

        if data_set is not None:
            if not isinstance(data_set, np.ndarray):
                data_set = np.array(data_set)
            if self.__k > data_set.shape[0]:
                raise ValueError('Parameter "k" user input is greater than volume of data set!')
        self.__data_set = data_set

        if labels is not None:
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            if self.__data_set is not None and self.__data_set.shape[0] != labels.shape[0]:
                raise ValueError('Lengths of data set and labels are not equal!')
            if self.__k > labels.shape[0]:
                raise ValueError('Parameter "k" user input is greater than volume of data set!')
        self.__labels = labels

    @property
    def k(self):
        return self.__k

    @property
    def data_set(self):
        return self.__data_set

    @property
    def labels(self):
        return self.__labels

    @k.setter
    def k(self, n):

        if not isinstance(n, int):
            if isinstance(n, float):
                warnings.warn('Type of parameter "n" is float, will transfer to an int.')
                n = int(n)
            else:
                raise ValueError(f'Type of parameter "k" must be numeric instead of {type(n)}.')
        if self.__data_set and n > self.__data_set.shape[0]\
                or self.__labels and n > self.__labels.shape[0]:
            raise ValueError("Parameter 'n' user input is greater than volume of data set!")

        self.__k = n

    @data_set.setter
    def data_set(self, ds):
        if not isinstance(ds, np.ndarray):
            ds = np.array(ds)
        if self.__k > ds.shape[0]:
            raise ValueError('Length of Parameter "ds" user input is lower than k!')
        if self.__labels and ds.shape[0] != self.__labels.shape[0]:
            raise ValueError("Length of Parameter 'ds' user input is not equal to labels'!")
        self.__data_set = ds

    @labels.setter
    def labels(self, ls):
        if not isinstance(ls, np.ndarray):
            ls = np.array(ls)
        if self.__data_set and self.__data_set.shape[1] != ls.shape[0]:
            raise ValueError("Length of parameter 'ls' user input is not equal to data set's.")
        if self.__k > ls.shape[0]:
            raise ValueError("Length of parameter 'ls' user input is lower than k!")
        self.__labels = ls

    def predict(self, inx):

        if self.__data_set is None:
            raise ValueError("Parameter data_set is None!")
        if self.__labels is None:
            raise ValueError("Parameter labels is None!")
        if not isinstance(inx, np.ndarray):
            inx = np.array(inx)
        if inx.shape[0] != self.__data_set.shape[1]:
            raise ValueError("Lengths of inx and data set are not equal!")

        diff_mat = (inx - self.__data_set) ** 2
        distances = diff_mat.sum(axis=1) ** 0.5
        sorted_index = distances.argsort()[:self.__k]
        labelist = [self.__labels[i] for i in sorted_index]
        label_dict = {key: labelist.count(key) for key in set(labelist)}
        labelist = sorted(label_dict.items(), key=lambda d:d[1], reverse=True)

        return labelist[0][0]


if __name__ == '__main__':

    x = np.array([1.0, 0])
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    label_list = np.array(["A", "A", 'B', "B"])
    knn = KNN(data_set=group, labels=label_list)
    print(knn.predict(x))