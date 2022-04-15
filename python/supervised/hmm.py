#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import joblib
import numpy as np


# 前向概率
def forward_proba(t, i, hmm, sequence):

    index = hmm.hidden_states[i]

    forwards = np.multiply(hmm.init_states, hmm.emit_mat[:, hmm.symbols[sequence[0]]].T)
    forwards = np.mat(forwards)
    for step in range(1, t):
        forwards *= hmm.trans_mat
        forwards = np.multiply(forwards, hmm.emit_mat[:, hmm.symbols[sequence[step]]].T)

    return forwards.A[0][index]


# 前向算法
def forward(hmm, sequence):
    T = len(sequence)

    forwards = np.multiply(hmm.init_states, hmm.emit_mat[:, hmm.symbols[sequence[0]]].T)
    yield forwards
    forwards = np.mat(forwards)
    for step in range(1, T):
        forwards = forwards * hmm.trans_mat
        forwards = np.multiply(forwards, hmm.emit_mat[:, hmm.symbols[sequence[step]]].T)
        yield forwards.A[0]


# 后向概率
def backward_proba(t, i, hmm, sequence):

    index = hmm.hidden_states[i]

    backwards = np.mat(np.ones(hmm.init_states.shape[0]))
    T = len(sequence)
    for step in range(T-1, t-1, -1):
        backwards = np.multiply(backwards, hmm.emit_mat[:, hmm.symbols[sequence[step]]])
        backwards = (hmm.trans_mat * backwards.T).T

    return backwards.A[0][index]


# 后向算法
def backward(hmm, sequence):
    T = len(sequence)

    backwards = np.mat(np.ones(hmm.init_states.shape[0]))
    yield backwards.A[0]

    for step in range(T-1, 0, -1):
        backwards = np.multiply(backwards, hmm.emit_mat[:, hmm.symbols[sequence[step]]])
        backwards = (hmm.trans_mat * backwards.T).T
        yield backwards.A[0]


# Baum-Welch算法（前向-后向算法）
def baum_welch(hmm, sequence):
    forwards_mat = np.mat(list(forward(hmm, sequence)))
    backwards_mat = np.mat(list(backward(hmm, sequence))[::-1])

    # E setp
    gamma = np.multiply(forwards_mat, backwards_mat)
    gamma /= gamma.sum(1)

    zeta = np.array([np.multiply(
        np.multiply(np.multiply(alpha.T, hmm.trans_mat),
                    np.array([emit[hmm.symbols[sequence[t + 1]]]
                              for emit in hmm.emit_mat])),
        backwards_mat[t + 1]
    )
        for t, alpha in enumerate(forwards_mat[:-1])])
    zeta = np.array([e / e.sum() for e in zeta])

    # M step
    hmm.init_states = gamma[0].A[0]
    hmm.trans_mat = zeta.sum(0) / gamma[:-1].sum(0).T
    ks = np.array([hmm.symbols[e] for e in sequence])
    jk = np.array([gamma[ks == i, :].sum(0).A[0] for i in hmm.symbols.values()])
    hmm.emit_mat = (jk / gamma.sum(0)).T


# 维特比算法
def viterbi(hmm, sequence):
    T = len(sequence)
    states_index = {v: k for k, v in hmm.hidden_states.items()}

    deltas = np.mat(hmm.init_states * hmm.emit_mat[:, hmm.symbols[sequence[0]]]).T
    psi = []
    for step in range(1, T):
        trans = np.multiply(deltas, hmm.trans_mat)
        deltas = np.max(trans, axis=0)
        deltas = np.multiply(deltas, hmm.emit_mat[:, hmm.symbols[sequence[step]]]).T
        index = np.argmax(trans, axis=0).A[0]
        psi.append(index)

    i = np.argmax(deltas)
    res = [states_index[i]]
    for arr in psi[::-1]:
        res.append(states_index[arr[i]])
        i = arr[i]

    return res[::-1]


def softmax(arr):
    arr = np.exp(arr)
    return arr / sum(arr)


# 初始化概率矩阵
def init_proba_matrix(m, n=None, mode='uniform'):

    if mode not in {'uniform', 'random'}:
        raise ValueError('Only "uniform" or "random" are allowed set parameter "mode".')
    if not isinstance(m, int):
        raise ValueError('Input "m" should be an int number!')
    if n is None or not isinstance(n, int):
        n = m

    if mode == 'random':
        mat = softmax(np.random.rand(m, n))
    else:
        mat = np.empty((m, n))
        mat.fill(1/n)

    return mat


# 隐马尔可夫模型
class HMM(object):

    def __init__(self, hidden_states=None, symbols=None, init_mode='uniform',
                 n_hidden_state=1, n_symbol=1, iter_num=1000):

        # 隐藏状态数目
        self.n_hidden_state = n_hidden_state
        # 观测符号数目
        self.n_symbol = n_symbol

        # 状态转移矩阵
        self.trans_mat = init_proba_matrix(self.n_hidden_state,
                                           mode=init_mode)
        self.trans_mat = np.mat(self.trans_mat)

        # 仿射矩阵
        self.emit_mat = init_proba_matrix(self.n_hidden_state,
                                          self.n_symbol,
                                          mode=init_mode)
        self.emit_mat = np.mat(self.emit_mat)

        # 初始状态向量
        self.init_states = init_proba_matrix(1, self.n_hidden_state,
                                             mode=init_mode)[0]

        # 隐状态集
        self.hidden_states = hidden_states
        # 观测符号集
        self.symbols = symbols

        self.iter_num = iter_num

    def fit(self, sequence):
        for _ in range(self.iter_num):
            baum_welch(self, sequence)

    def predict(self, sequence):
        return viterbi(self, sequence)

    def save(self, fpath):
        joblib.dump(self, fpath)


if __name__ == '__main__':

    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    pai = np.array([0.2, 0.4, 0.4])
    seq = (1, 2, 1)

    hmm = HMM()
    hmm.trans_mat = A
    hmm.emit_mat = B
    hmm.init_states = pai
    hmm.hidden_states = {1: 0, 2: 1, 3: 2}
    hmm.symbols = {'红': 0, '白': 1}
    sent = ('红', '白', '红')
