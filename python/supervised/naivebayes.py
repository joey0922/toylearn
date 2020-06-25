#! /usr/env/bin python
# -*- encoding:utf-8 -*-

import numpy as np

'''
优点：在数据较少的情况下仍然有效；可以处理多类别问题；实现简单，学习与预测效率都很高
缺点：对于输入数据的准备方式比较敏感；分类性能不一定跟高
'''


# simple implement of naive bayes algorithm
class NaiveBayes:

    def __init__(self, lp=1):

        if not isinstance(lp, (int, float)) or lp < 0:
            raise ValueError("Parameter 'lp' must be a not negative numeric.")
        self.__lp = lp  # 贝叶斯估计平滑系数

        self.__label_proba = None  # 类别的先验概率的对数
        self.__conditional_proba = None  # 特征的条件概率的对数

    @property
    def lp(self):
        return self.__lp

    @property
    def label_probability(self):
        return self.__label_proba

    @property
    def conditional_proba(self):
        return self.__conditional_proba

    @lp.setter
    def lp(self, p):
        if not isinstance(p, (int, float)) or p < 0:
            raise ValueError("Parameter 'p' must be a not negative numeric.")
        self.__lp = p

    def fit(self, train_set, labels):

        # 检查参数
        if not isinstance(train_set, np.ndarray):
            train_set = np.array(train_set)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        # 计算类别的先验概率
        train_cnt = labels.shape[0]  # 训练集数量
        # 每个类别的数量字典
        label_cnt = {la: labels[labels == la].shape[0] for la in set(labels)}
        k = len(set(labels))  # 类别数目
        # 类别先验概率的对数
        self.__label_proba = {la: np.log((label_cnt[la]+self.__lp)/(train_cnt+k*self.__lp))
                              for la in label_cnt.keys()}

        # 计算特征条件概率
        fea_cat = np.array([set(train_set[:, i]) for i, _ in enumerate(train_set[0])])
        fea_cnt = np.array([len(set(train_set[:, i])) for i, _ in enumerate(train_set[0])])
        self.__conditional_proba = {lb: tuple({fea: np.log(((train_set[
                                                                 np.where(labels == lb)[0]]
                                                             [:, i] == fea).sum()
                                                            + self.__lp)
                                                           / (lcnt+self.__lp*fea_cnt[i]))
                                               for fea in fset}
                                              for i, fset in enumerate(fea_cat))
                                    for lb, lcnt in label_cnt.items()}

        return self

    def predict(self, raw_x):

        if not isinstance(raw_x, np.ndarray):
            raw_x = np.array(raw_x)

        label_list = tuple(self.__label_proba.keys())
        proba_arr = np.array([sum(self.__conditional_proba[lb][j].get(key, 0)
                                  for j, key in enumerate(raw_x)) + self.__label_proba[lb]
                              for _, lb in enumerate(label_list)])
        idx = proba_arr.argmax(axis=0)
        label = label_list[idx]

        return label


if __name__ == '__main__':

    X = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'],
                  [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'],
                  [3, 'M'], [3, 'L'], [3, 'L']])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    nb = NaiveBayes(lp=1)
    model = nb.fit(X, y)
    raw = np.array([2, 'S'])
    print(model.predict(raw))