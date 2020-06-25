# -*- encoding:utf-8 -*-

# KMP算法
'''
假设现在文本串S匹配到 i 位置，模式串P匹配到 j 位置
如果j = -1，或者当前字符匹配成功（即S[i] == P[j]），都令i++，j++，继续匹配下一个字符；
如果j != -1，且当前字符匹配失败（即S[i] != P[j]），则令 i 不变，j = next[j]。
此举意味着失配时，模式串P相对于文本串S向右移动了j - next [j] 位。
换言之，当匹配失败时，模式串向右移动的位数为：
失配字符所在位置 - 失配字符对应的next值，即移动的实际位数为：j - next[j]，且此值大于等于1
'''
def get_next(pattern):

    size = len(pattern)
    nxt = [-1 if i == 0 else 0 for i in range(size)]
    if size <= 2:
        return nxt

    k, j = -1, 0
    while j < size - 1:
        if k == -1 or pattern[k] == pattern[j]:
            k += 1
            j += 1
            nxt[j] = k if pattern[j] != pattern[k] else nxt[k]
        else:
            k = nxt[k]

    return nxt

def KMP(string, pattern):

    slen = len(string)
    plen = len(pattern)

    if slen == 0 or plen == 0:
        return

    nxt = get_next(pattern)
    i, j = 0, 0
    while i < slen and j < plen:
        if j == -1 or string[i] == pattern[j]:
            i += 1
            j += 1
        else:
            j = nxt[j]

    if j == plen:
        return i - j
    else:
        return -1

if __name__ == '__main__':

    string = ''
    pattern = 'ABCDABCE'
    nxt = get_next(pattern)
    print(nxt)