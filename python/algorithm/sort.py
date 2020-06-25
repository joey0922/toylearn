# -*- encoding:utf-8 -*-

import numpy as np


#生成随机数组
def rand_arr(low, high, size):
    return np.random.random_integers(low, high, size)


'''
冒泡排序：
1.比较相邻的元素。如果第一个比第二个大，就交换他们两个。
2.对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。
3.针对所有的元素重复以上的步骤，除了最后一个。
4.持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。
'''
def bubble_sort(arr):

    size = len(arr)

    for i in range(size):
        for j in range(size-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
            # print(f'第{i+1}轮第{j+1}次结果:{arr}')
        # print(f'第{i+1}轮结果:{arr}')

    return arr

'''
选择排序：
1.首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置
2.再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。
3.重复第二步，直到所有元素均排序完毕。
'''
def selection_sort(arr):

    size = len(arr)

    for i in range(size-1):
        index = i
        for j in range(i+1, size):
            if arr[index] > arr[j]:
                index = j
        if index != i:
            arr[i], arr[index] = arr[index], arr[i]
        # print(f'第{i+1}轮结果:{arr}')

    return arr

'''
插入排序：
1.将第一待排序序列第一个元素看做一个有序序列，把第二个元素到最后一个元素当成是未排序序列。
2.从头到尾依次扫描未排序序列，将扫描到的每个元素插入有序序列的适当位置。
（如果待插入的元素与有序序列中的某个元素相等，则将待插入元素插入到相等元素的后面。）
'''
def insertion_sort(arr):

    size = len(arr)

    for i in range(1, size):
        index = i - 1
        current = arr[i]
        while index >= 0 and arr[index] > current:
            arr[index+1] = arr[index]
            index -= 1
        arr[index+1] = current
        # print(f'第{i}轮结果:{arr}')

    return arr

'''
希尔排序：也称递减增量排序算法，是插入排序的一种更高效的改进版本。但希尔排序是非稳定排序算法
基本思想：先将整个待排序的记录序列分割成为若干子序列分别进行直接插入排序，待整个序列中的记录'基本有序'时，
再对全体记录进行依次直接插入排序。
1.选择一个增量序列 t1，t2，……，tk，其中 ti > tj, tk = 1；
2.按增量序列个数 k，对序列进行 k 趟排序；
3.每趟排序，根据对应的增量 ti，将待排序列分割成若干长度为 m 的子序列，分别对各子表进行直接插入排序。
仅增量因子为 1 时，整个序列作为一个表来处理，表长度即为整个序列的长度。
'''
def shell_sort(arr):

    size = len(arr)

    gap = 1
    while gap < size // 3:
        gap = gap * 3 + 1

    while gap > 0:
        for i in range(gap, size):
            j = i - gap
            current = arr[i]
            while j >= 0 and arr[j] > current:
                arr[j+gap] = arr[j]
                j -= gap
            arr[j+gap] = current
        gap = gap // 3

    return arr

'''
归并排序：采用分治法，始终都是 O(nlogn) 的时间复杂度，代价是需要额外的内存空间
1.申请空间，使其大小为两个已经排序序列之和，该空间用来存放合并后的序列；
2.设定两个指针，最初位置分别为两个已经排序序列的起始位置；
3.比较两个指针所指向的元素，选择相对小的元素放入到合并空间，并移动指针到下一位置；
4.重复步骤 3 直到某一指针达到序列尾；
5.将另一序列剩下的所有元素直接复制到合并序列尾。
'''
def merge(left, right):

    result = []
    if not isinstance(left, list):
        left = list(left)
    if not isinstance(right, list):
        right = list(right)

    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))

    if left:
        result.extend(left)
    if right:
        result.extend(right)

    result = np.array(result)

    return result

def merge_sort(arr):

    size = len(arr)

    if size < 2:
        return arr

    middle = size // 2
    left, right = arr[:middle], arr[middle:]

    return merge(merge_sort(left), merge_sort(right))

'''
快速排序：
1.从数列中挑出一个元素，称为 “基准”（pivot）;
2.重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。
在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；
3.递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序；
'''
def quick_sort(arr, left=None, right=None):

    left = 0 if not isinstance(left, (int, float)) else left
    right = len(arr) - 1 if not isinstance(right, (int, float)) else right

    if left >= right:
        return

    key = arr[left]
    l = left
    r = right
    while l < r:

        while l < r and arr[r] >= key:
            r -= 1
        arr[l] = arr[r]

        while l < r and arr[l] <= key:
            l += 1
        arr[r] = arr[l]

    arr[l] = key
    quick_sort(arr, left, l-1)
    quick_sort(arr, l+1, right)

    return arr

'''
堆排序：
# 1.创建一个堆 H[0……n-1]；
# 2.把堆首（最大值）和堆尾互换；
# 3.把堆的尺寸缩小 1，并调用 shift_down(0)，目的是把新的数组顶端数据调整到相应位置；
# 4.重复步骤 2，直到堆的尺寸为 1。
1.首先将待排序的数组构造成一个大根堆，此时，整个数组的最大值就是堆结构的顶端
2.将顶端的数与末尾的数交换，此时，末尾的数为最大值，剩余待排序数组个数为n-1
3.将剩余的n-1个数再构造成大根堆，再将顶端数与n-1位置的数交换，如此反复执行，便能得到有序数组
'''
#构造成大根堆
def heapify(arr, index, size):

    left = 2 * index + 1
    right = 2 * index + 2
    largest_idx = index

    if left < size and arr[largest_idx] < arr[left]:
        largest_idx = left

    if right < size and arr[largest_idx] < arr[right]:
        largest_idx = right

    # 父结点不是最大值，与孩子中较大的值交换
    if index != largest_idx:
        arr[index], arr[largest_idx] = arr[largest_idx], arr[index]
        heapify(arr, largest_idx, size)

def heap_sort(arr):

    size = len(arr)

    #构建大根堆
    for i in range(size//2-1, -1, -1):
        heapify(arr, i, size)

    #一个个交换元素
    for i in range(size-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, 0, i)

    return arr

'''
计数排序：
1.根据待排序集合中最大元素和最小元素的差值范围，申请额外空间；
2.遍历待排序集合，将每一个元素出现的次数记录到元素值对应的额外空间内；
3.对额外空间内数据进行计算，得出每一个元素的正确位置；
4.将待排序集合每一个元素移动到计算得出的正确位置上。
'''
def count_sort(arr):

    min_val, max_val = min(arr), max(arr)
    size = max_val - min_val + 1

    #创建额外空间
    count = np.zeros(size)
    for i in arr:
        count[i-min_val] += 1

    #数组排序
    arr = []
    for i, num in enumerate(count):
        for j in range(int(num)):
            arr.append(i+min_val)

    arr = np.array(arr)
    return arr

'''
基数排序：
1.取得数组中的最大数,并取得位数
2.arr为原始数组,从最低位开始取每个位组成radix数组
3.对radix进行计数排序(利用计数排序适用于小范围数的特点)
'''
def radix_sort(arr):

    if not isinstance(arr, list):
        arr = list(arr)

    max_val = max(arr)
    size = len(str(max_val))

    for i in range(size):
        bucket_list = [[] for _ in range(10)]
        for n in arr:
            bucket_list[(n//10**i) % 10].append(n)

        print(bucket_list, end=' ')
        arr.clear()
        for l in bucket_list:
            for n in l:
                arr.append(n)
        print(arr)

    arr = np.array(arr)
    return arr

if __name__ == '__main__':

    # arr = rand_arr(1, 100, 10)
    arr = np.array([5, 40, 29, 26, 58, 13, 54, 97, 36, 31])
    print(f'原始数组：{arr}')
    # res = bubble_sort(arr)
    # res = selection_sort(arr)
    # res = insertion_sort(arr)
    # res = shell_sort(arr)
    # res = merge_sort(arr)
    res = quick_sort(arr)
    # res = heap_sort(arr)
    # res = count_sort(arr)
    # res = radix_sort(arr)
    print(f'最终排序结果：{res}')