# -*- encoding:utf-8 -*-

'''
二分查找：
1.将表中间位置记录的关键字与查找关键字比较，如果两者相等，则查找成功；
2.否则利用中间位置记录将表分成前、后两个子表，如果中间位置记录的关键字大于查找关键字，则进一步查找前一子表，
否则进一步查找后一子表。
3.重复以上过程，直到找到满足条件的记录，使查找成功，或直到子表不存在为止，此时查找不成功。
复杂度分析
    时间复杂度：折半搜索每次把搜索区域减少一半，时间复杂度为 O(logn)
    空间复杂度：O(1)
'''
#循环二分查找
def binary_search(arr, target):

    left, right = 0, len(arr)-1

    while left <= right:
        index = (left + right) // 2
        if arr[index] < target:
            left = index + 1
        elif arr[index] > target:
            right = index - 1
        else:
            return index

    return None

#递归二分查找
def recur_bin_search(arr, target, left=None, right=None):

    left = left if isinstance(left, int) else 0
    right = right if isinstance(right, int) else len(arr)-1

    index = (left + right) // 2

    if arr[index] == target:
        return index
    elif left == right:
        return None
    else:
        if target < arr[index]:
            right = index-1
        else:
            left = index+1
        return recur_bin_search(arr, target, left, right)

'''
插值查找：
算法简介
    插值查找是根据要查找的关键字key与查找表中最大最小记录的关键字比较后的查找方法，其核心就在于插值的计算公式 (high-low)*(key-a[low])/(a[high]-a[low])。
    时间复杂度o(logn)但对于表长较大而关键字分布比较均匀的查找表来说，效率较高。
算法思想
    基于二分查找算法，将查找点的选择改进为自适应选择，可以提高查找效率。当然，差值查找也属于有序查找。
    注：对于表长较大，而关键字分布又比较均匀的查找表来说，插值查找算法的平均性能比折半查找要好的多。反之，数组中如果分布非常不均匀，那么插值查找未必是很合适的选择。
复杂度分析 
    时间复杂性：如果元素均匀分布，则O（log log n）），在最坏的情况下可能需要 O（n）。
    空间复杂度：O（1）。
'''
#循环插值查找
def interpo_search(arr, target):

    left, right = 0, len(arr) - 1

    while left < right:
        step = (right - left) * (target - arr[left]) // (arr[right] - arr[left])
        index = left + step if step else (left + right) // 2
        # index = left + step
        if target < arr[index]:
            right = index - 1
        elif target > arr[index]:
            left = index + 1
        else:
            return index

    return None

'''
斐波那契查找：
算法简介
    斐波那契数列，又称黄金分割数列，指的是这样一个数列：1、1、2、3、5、8、13、21、····，
    在数学上，斐波那契被递归方法如下定义：F(1)=1，F(2)=1，F(n)=f(n-1)+F(n-2) （n>=2）。
    该数列越往后相邻的两个数的比值越趋向于黄金比例值（0.618）。
算法描述 
    斐波那契查找就是在二分查找的基础上根据斐波那契数列进行分割的。
    在斐波那契数列找一个等于略大于查找表中元素个数的数F[n]，
    将原查找表扩展为长度为F[n](如果要补充元素，则补充重复最后一个元素，直到满足F[n]个元素)，
    完成后进行斐波那契分割，即F[n]个元素分割为前半部分F[n-1]个元素，后半部分F[n-2]个元素，
    找出要查找的元素在那一部分并递归，直到找到。
复杂度分析 
    最坏情况下，时间复杂度为O(log2n)，且其期望复杂度也为O(log2n)。
'''
# 斐波那契数列
# def fibonacci(n):
#     return 1 if n == 1 or n == 2 else fibonacci(n-1) + fibonacci(n-2)

# def fibonacci(n):
#
#     a, b = 0, 1
#     for i in range(1, n+1):
#         a, b = b, a+b
#         yield a

def fibonacci(bound):

    a, b = 0, 1

    while True:
        a, b = b, a + b
        yield a
        if a > bound:
            break

def fib_search(arr, target):

    left, right = 0, len(arr) - 1

    if not isinstance(arr, list):
        arr = list(arr)

    # 需要一个现成的斐波那契列表。其最大元素的值必须超过查找表中元素个数的数值。
    fib = [i for i in fibonacci(right+1)]

    # 为了使得查找表满足斐波那契特性，在表的最后添加几个同样的值
    # 这个值是原查找表的最后那个元素的值
    # 添加的个数由F[k]-1-right决定
    k = 0
    while right > fib[k] - 1: k += 1
    arr.extend([arr[right] for _ in range(fib[k] - right - 1)])
    # i = right
    # while fib[k] - 1 > i:
    #     arr.append(arr[right])
    #     i += 1

    '''
    算法主逻辑
    1）当target=a[index]时，查找成功；
    2）当target<a[index]时，新的查找范围是第left个到第index-1个，此时范围个数为fib[k-1] - 1个，
    即数组左边的长度，所以要在[left, fib[k - 1] - 1]范围内查找；
    3）当target>a[index]时，新的查找范围是第index+1个到第right个，此时范围个数为fib[k-2] - 1个，
    即数组右边的长度，所以要在[fib[k - 2] - 1]范围内查找。
    '''
    while left <= right:

        # 为了防止fib列表下标溢出
        index = left if k < 2 else left + fib[k-1] - 1

        if target < arr[index]:
            right = index - 1
            k -= 1
        elif target > arr[index]:
            left = index + 1
            k -= 2
        else:
            return index if index <= right else right

    return None


#树表查找算法
'''
1.二叉树查找算法
算法简介
    二叉查找树是先对待查找的数据进行生成树，确保树的左分支的值小于右分支的值，
    然后在就行和每个节点的父节点比较大小，查找最适合的范围。 
    这个算法的查找效率很高，但是如果使用这种查找方法要首先创建树。 
算法思想
    二叉查找树（BinarySearch Tree）或者是一棵空树，或者是具有下列性质的二叉树：
    1）若任意节点的左子树不空，则左子树上所有结点的值均小于它的根结点的值；
    2）若任意节点的右子树不空，则右子树上所有结点的值均大于它的根结点的值；
    3）任意节点的左、右子树也分别为二叉查找树。
复杂度分析
    它和二分查找一样，插入和查找的时间复杂度均为O(logn)，但是在最坏的情况下仍然会有O(n)的时间复杂度。
    原因在于插入和删除元素的时候，树没有保持平衡。
'''
#二叉树节点类
class BTNode(object):

    def __init__(self, data, left=None, right=None):
        '''
        :param data:节点储存的数据
        :param left:节点左子树
        :param right:节点右字树
        '''
        super(BTNode, self).__init__()
        self.__data = data
        self.__left = left
        self.__right = right

    @property
    def data(self):
        return self.__data

    @property
    def left(self):
        return self.__left

    @property
    def right(self):
        return self.__right

    @data.setter
    def data(self, d):
        if not isinstance(d, (int, float)) and not isinstance(d.data, (int, float)):
            raise ValueError('type of data input is not int or float')

        self.__data = d

    @left.setter
    def left(self, l):
        if not isinstance(l, (int, float)) and not isinstance(l.data, (int, float)):
            raise ValueError('type of left input is not int or float')

        self.__left = l

    @right.setter
    def right(self, r):
        if not isinstance(r, (int, float)) and not isinstance(r.data, (int, float)):
            raise ValueError('type of right input is not int or float')

        self.__right = r


#基于BTNode类的二叉查找树
class BinaryTree(object):

    def __init__(self):
        super(BinaryTree, self).__init__()
        self.__root = None

    def is_empty(self):
        return self.__root is None

    def search(self, target):
        '''
        :param target: 关键码
        :return:查询节点或None
        '''

        bt = self.__root
        while bt:
            entry = bt.data
            if target < entry:
                bt = bt.left
            elif target > entry:
                bt = bt.right
            else:
                return entry

        return None

    def insert(self, target):
        '''
        :param target:关键码
        :return:布尔值
        '''

        bt = self.__root
        if not bt:
            self.__root = BTNode(target)
            return
        while True:
            entry = bt.data
            if target < entry:
                if bt.left is None:
                    bt.left = BTNode(target)
                    return
                bt = bt.left
            elif target > entry:
                if bt.right is None:
                    bt.right = BTNode(target)
                    return
                bt = bt.right
            else:
                bt.data = target
                return

    def delete(self, target):
        '''
        :param target:关键码
        :return:布尔值
        '''

        # 维持father为child的父节点，用于后面的链接操作
        father, child = None, self.__root
        if not child:
            print('It is a empty tree!')
            return

        while child and child.data != target:

            father = child

            if target < child.data:
                child = child.left
            else:
                child = child.right
            # 当树中没有关键码target时，结束退出
            if not child:
                return

        # 上面已将找到了要删除的节点，用child引用。而father则是child的父节点或者None（child为根节点时）
        if not child.left:
            if father is None:
                self.__root = child.right
            elif child is father.left:
                father.left = child.right
            else:
                father.right = child.right
            return

        # 查找节点child的左子树的最右节点，将child的右子树链接为该节点的右子树
        # 该方法可能会增大树的深度，效率并不算高。可以设计其它的方法。
        r = child.left
        while r.right:
            r = r.right
        r.right = child.right
        if father is None:
            self.__root = child.left
        elif father.left is child:
            father.left = child.left
        else:
            father.right = child.left

    # 实现二叉树的中序遍历算法, 展示我们创建的二叉查找树。直接使用python内置的列表作为一个栈。
    def __iter__(self):

        stack = []
        node = self.__root

        while node or stack:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.data
            node = node.right

#二叉树搜索
def bintree_search(arr, target):

    btree = BinaryTree()
    for i in range(len(arr)):
        btree.insert(arr[i])

    btree.delete(target)

    for i in btree:
        print(i, end=' ')


'''
2.平衡查找树之2-3查找树（2-3 Tree）
2-3查找树定义：
    2-3树运行每个节点保存1个或者两个的值。对于普通的2节点(2-node)，
    他保存1个key和左右两个自己点。对应3节点(3-node)，保存两个Key，2-3查找树的定义如下： 
　　 1）要么为空，要么： 
　　 2）对于2节点，该节点保存一个key及对应value，以及两个指向左右节点的节点，
       左节点也是一个2-3节点，所有的值都比key要小，右节点也是一个2-3节点，所有的值比key要大。 
　　 3）对于3节点，该节点保存两个key及对应value，以及三个指向左中右的节点。
       左节点也是一个2-3节点，所有的值均比两个key中的最小的key还要小；
       中间节点也是一个2-3节点，中间节点的key值在两个跟节点key值之间；
       右节点也是一个2-3节点，节点的所有key值比两个key中的最大的key还要大。
2-3查找树的性质：
　　 1）如果中序遍历2-3查找树，就可以得到排好序的序列； 
　　 2）在一个完全平衡的2-3查找树中，根节点到每一个为空节点的距离都相同。
      （这也是平衡树中“平衡”一词的概念，根节点到叶节点的最长距离对应于查找算法的最坏情况，
       而平衡树中根节点到叶节点的距离都一样，最坏情况也具有对数复杂度。） 
复杂度分析：
    2-3树的查找效率与树的高度是息息相关的。
    1）在最坏的情况下，也就是所有的节点都是2-node节点，查找效率为logN
    2）在最好的情况下，所有的节点都是3-node节点，查找效率为log3N约等于0.631logN
　　距离来说，对于1百万个节点的2-3树，树的高度为12-20之间，对于10亿个节点的2-3树，树的高度为18-30之间。
　　对于插入来说，只需要常数次操作即可完成，因为他只需要修改与该节点关联的节点即可，
    不需要检查其他节点，所以效率和查找类似。
'''
# 23树节点
class TTNode(object):

    def __init__(self, key):
        super(TTNode, self).__init__()
        self.key1 = key
        self.key2 = None
        self.left = None
        self.middle = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.middle is None and self.right is None

    def is_full(self):
        return self.key2 is not None

    def has_key(self, k):
        return True if self.key1 == k or (self.key2 is not None and self.key2 == k) else False

    def get_child(self, k):

        if k < self.key1:
            return self.left
        elif self.key2 is None or k < self.key2:
            return self.middle
        else:
            return self.right

# 23树类
class TwoThreeTree(object):

    def __init__(self):
        super(TwoThreeTree, self).__init__()
        self.root = None

    def get(self, key):
        return None if self.root is None else self._get(self.root, key)

    def _get(self, node, key):

        if node is None:
            return None
        elif node.has_key(key):
            return node
        else:
            child = node.get_child(key)
            return self._get(child, key)

    def put(self, key):

        if self.root is None:
            self.root = TTNode(key)
        else:
            pkey, pref = self._put(self.root, key)
            if pkey is not None:
                newnode = TTNode(pkey)
                newnode.left = self.root
                newnode.middle = pref
                self.root = newnode

    def _put(self, node, key):

        if node.has_key(key):
            return None, None
        elif node.is_leaf():
            return self._add2node(node, key, None)
        else:
            child = node.get_child(key)
            pkey, pref = self._put(child, key)
            if pkey is None:
                return None, None
            else:
                return self._add2node(node, pkey, pref)

    def _add2node(self, node, key, pref):

        if node.is_full():
            return self._split_node(node, key, pref)
        else:
            if key < node.key1:
                node.key2 = node.key1
                node.key1 = key
                if pref is not None:
                    node.right = node.middle
                    node.middle = pref
            else:
                node.key2 = key
                if pref is not None:
                    node.right = pref

            return None, None

    def _split_node(self, node, key, pref):

        newnode = TTNode(None)
        if key < node.key1:
            pkey = node.key1
            node.key1 = key
            newnode.key1 = node.key2
            if pref is not None:
                newnode.left = node.middle
                newnode.middle = node.right
                node.right = pref
        elif key < node.key2:
            pkey = key
            newnode.key1 = node.key2
            if pref is not None:
                newnode.left = pref
                newnode.middle = node.right
        else:
            pkey = node.key2
            newnode.key1 = key
            if pref is not None:
                newnode.left = node.right
                newnode.middle = pref
        node.key2 = None

        return pkey, newnode

'''
3.平衡查找树之红黑树（Red-Black Tree）
基本思想：
    红黑树的思想就是对2-3查找树进行编码，尤其是对2-3查找树中的3-nodes节点添加额外的信息。
    红黑树中将节点之间的链接分为两种不同类型，红色链接，他用来链接两个2-nodes节点来表示一个3-nodes节点。
    黑色链接用来链接普通的2-3节点。
    特别的，使用红色链接的两个2-nodes来表示一个3-nodes节点，并且向左倾斜，
    即一个2-node是另一个2-node的左子节点。这种做法的好处是查找的时候不用做任何修改，
    和普通的二叉查找树相同。
红黑树的定义：
　　红黑树是一种具有红色和黑色链接的平衡查找树，一种简单实现2-3树的数据结构，同时满足： 
　　1）红色节点向左倾斜 ；
　　2）一个节点不可能有两个红色链接；
    3）整个树完全黑色平衡，即从根节点到所有叶子结点的路径上，黑色链接的个数都相同。
红黑树的性质：
    整个树完全黑色平衡，即从根节点到所有叶子结点的路径上，
    黑色链接的个数都相同（2-3树的第2）性质，从根节点到叶子节点的距离都相等。
复杂度分析：
    最坏的情况就是，红黑树中除了最左侧路径全部是由3-node节点组成，
    即红黑相间的路径长度是全黑路径长度的2倍。
红黑树的平均高度大约为logN 
'''
from random import randint

RED = 'red'
BLACK = 'black'

# 红黑树节点
class RBNode(object):

    def __init__(self, val):
        super(RBNode, self).__init__()
        self.val = val
        self.left = None
        self.middle = None
        self.parent = None

    def paint(self, color):
        self.color = color

# 左旋
def levo(b, c):

    a = b.parent
    a.left = c
    c.parent = a
    b.parent = c
    c.left = b

# 红黑树类
class RBTree(object):

    def __init__(self):
        super(RBTree, self).__init__()
        self.root = None
        self.zlist = []

    def left_rotate(self, xnode):

        ynode = xnode.right
        if ynode is None:
            # 右节点为空，不旋转
            return
        else:
            beta = ynode.left
            xnode.right = beta
            if beta is not None:
                beta.parent = xnode

            p = xnode.parent
            ynode.parent = p
            if p is None:
                # xnode原来是root
                self.root = ynode
            elif xnode == p.left:
                p.left = ynode
            else:
                p.right = ynode
            ynode.left = xnode
            xnode.parent = ynode

    def right_rotate(self, ynode):

        # ynode是一个节点
        xnode = ynode.left
        if xnode is None:
            # 右节点为空，不旋转
            return
        else:
            beta = xnode.right
            ynode.left = beta
            if beta is not None:
                beta.parent = ynode

            p = ynode.parent
            xnode.parent = p
            if p is None:
                # ynode原来是root
                self.root = xnode
            elif ynode == p.left:
                p.left = xnode
            else:
                p.right = xnode
            xnode.right = ynode
            ynode.parent = xnode

    def insert(self, val):

        znode = RBNode(val)
        ynode = None
        xnode = self.root

        while xnode is not None:
            ynode = xnode
            if znode.val < xnode.val:
                xnode = xnode.left
            else:
                xnode = xnode.right

        znode.paint(RED)
        znode.parent = ynode

        # 插入znode之前为空的RBNode
        if ynode is None:
            self.root = znode
            self.insert_fixup(znode)
            return

        if znode.val < ynode.val:
            ynode.left = znode
        else:
            ynode.right = znode

        # 如果znode的父节点ynode为红色，需要fixup，如果znode的父节点ynode为黑色，则不用调整
        if ynode.color == RED:
            self.insert_fixup(znode)
        else:
            return

    def insert_fixup(self, znode):

        # case 1:znode为root节点
        if znode.parent is None:
            znode.paint(BLACK)
            self.root = znode
            return

        # case 2:znode的父节点为黑色
        if znode.parent.color == BLACK:
            return

        # 下面的几种情况，都是znode.parent.color == RED:
        p = znode.parent
        g = p.parent
        if g is None:
            return
        if g.right == p:
            ynode = g.left
        else:
            ynode = g.right

        # case 3-0:znode没有叔叔。即：y为NIL节点，注意，此时znode的父节点一定是RED


'''
4.B树和B+树（B Tree/B+ Tree）
B树简介：
    B树可以看作是对2-3查找树的一种扩展，即他允许每个节点有M-1个子节点。
        1)根节点至少有两个子节点；
        2)每个节点有M-1个key，并且以升序排列；
        3)位于M-1和M key的子节点的值位于M-1 和M key对应的Value之间；
        4)非叶子结点的关键字个数=指向儿子的指针个数-1；
        5)非叶子结点的关键字：K[1], K[2], …, K[M-1]；且K[i] ；
        6)其它节点至少有M/2个子节点；
        7)所有叶子结点位于同一层；
B树算法思想：
    B-树的搜索，从根结点开始，对结点内的关键字（有序）序列进行二分查找，
    如果命中则结束，否则进入查询关键字所属范围的儿子结点；
    重复，直到所对应的儿子指针为空，或已经是叶子结点；
B树的特性：
    1.关键字集合分布在整颗树中；
    2.任何一个关键字出现且只出现在一个结点中；
    3.搜索有可能在非叶子结点结束；
    4.其搜索性能等价于在关键字全集内做一次二分查找；
    5.自动层次控制；
    由于限制了除根结点以外的非叶子结点，至少含有M/2个儿子，
    确保了结点的至少利用率，其最底搜索性能为O(LogN)
    
B+树简介：
    B+树是B-树的变体，也是一种多路搜索树：
        1.其定义基本与B-树同，除了：
        2.非叶子结点的子树指针与关键字个数相同；
        3.非叶子结点的子树指针P[i]，指向关键字值属于[K[i], K[i+1])的子树
        4.B-树是开区间；
        5.为所有叶子结点增加一个链指针；
        6.所有关键字都在叶子结点出现；
B+树算法思想：
    B+的搜索与B-树也基本相同，区别是B+树只有达到叶子结点才命中（B-树可以在非叶子结点命中），
    其性能也等价于在关键字全集做一次二分查找；
B+树的特性：
    1.所有关键字都出现在叶子结点的链表中（稠密索引），且链表中的关键字恰好是有序的；
    2.不可能在非叶子结点命中；
    3.非叶子结点相当于是叶子结点的索引（稀疏索引），叶子结点相当于是存储（关键字）数据的数据层；
    4.更适合文件索引系统；
'''
class BTree(object):

    def __init__(self):
        pass

'''
树表查找总结:
    二叉查找树平均查找性能不错，为O(logn)，但是最坏情况会退化为O(n)。
    在二叉查找树的基础上进行优化，我们可以使用平衡查找树。
    平衡查找树中的2-3查找树，这种数据结构在插入之后能够进行自平衡操作，
    从而保证了树的高度在一定的范围内进而能够保证最坏情况下的时间复杂度。
    但是2-3查找树实现起来比较困难，红黑树是2-3树的一种简单高效的实现，
    他巧妙地使用颜色标记来替代2-3树中比较难处理的3-node节点问题。
    红黑树是一种比较高效的平衡查找树，应用非常广泛，很多编程语言的内部实现都或多或少的采用了红黑树。
    除此之外，2-3查找树的另一个扩展——B/B+平衡树，在文件系统和数据库系统中有着广泛的应用。
'''

'''
分块查找
算法简介：
    要求是顺序表，分块查找又称索引顺序查找，它是顺序查找的一种改进方法。 
算法思想：
    将n个数据元素"按块有序"划分为m块（m≤n）。
    每一块中的结点不必有序，但块与块之间必须"按块有序"；
    即第1块中任一元素的关键字都必须小于第2块中任一元素的关键字；
    而第2块中任一元素又都必须小于第3块中的任一元素，……
算法流程：
    1、先选取各块中的最大关键字构成一个索引表；
    2、查找分两个部分：先对索引表进行二分查找或顺序查找，以确定待查记录在哪一块中；
    3、在已确定的块中用顺序法进行查找。
复杂度分析：
    时间复杂度：O(log(m)+N/m)
'''
def block_search(arr):
    pass

'''
哈希查找
算法简介：
    哈希表就是一种以键-值(key-indexed) 存储数据的结构，只要输入待查找的值即key，即可查找到其对应的值。
算法思想：
    哈希的思路很简单，如果所有的键都是整数，那么就可以使用一个简单的无序数组来实现：
    将键作为索引，值即为其对应的值，这样就可以快速访问任意键的值。
    这是对于简单的键的情况，我们将其扩展到可以处理更加复杂的类型的键。
算法流程：
　　1）用给定的哈希函数构造哈希表；
　　2）根据选择的冲突处理方法解决地址冲突；常见的解决冲突的方法：拉链法和线性探测法。
　　3）在哈希表的基础上执行哈希查找。
复杂度分析：
    单纯论查找复杂度：对于无冲突的Hash表而言，查找复杂度为O(1)
    （注意，在查找之前我们需要构建相应的Hash表）。
'''
class HashTable(object):

    def __init__(self, size):

        self.element = [None for _ in range(size)]
        self.count = size

    # 散列函数采用除留余数法
    def hash(self, key):
        return key % self.count

    # 插入关键字到哈希表内
    def insert(self, key):
        # 求散列地址
        address = self.hash(key)
        # 当前位置已经有数据了，发生冲突，线性探测下一地址是否可用
        while self.element[address]:
            address = (address + 1) % self.count
        # 没有冲突则直接保存
        self.element[address] = key

    # 查找关键字，返回布尔值
    def search(self, key):

        start = address = self.hash(key)
        while self.element[address] != key:
            address = (address + 1) % self.count
            # 说明没找到或者循环到了开始的位置
            if not self.element[address] or address == start:
                return False

        return True

def hash_search(arr, target):
    pass


if __name__ == '__main__':

    arr = [5, 13, 26, 29, 31, 36, 40, 54, 58, 97]
    target = 26
    # index = recur_bin_search(arr, target)
    # index = interpo_search(arr, target)
    # index = fib_search(arr, target)
    bintree_search(arr, target)
    # print(index)