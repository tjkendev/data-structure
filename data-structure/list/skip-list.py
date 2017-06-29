# encoding: utf-8

# 参考:
# - Wikipedia (Skip list): https://en.wikipedia.org/wiki/Skip_list
# - Pugh, William. A skip list cookbook. 1998.

import random
random.seed()

# SkipList (インデックスなしでk番目アクセスが線形オーダー)
class SkipList:
    def __init__(self, level):
        self.level = level
        self.rmax = 2**level-1
        self.root = [None] * (level+2)
        self.size = 0

    # 1 ≦ k ≦ level
    # Pr(X = k) = 2^{k-1}/(2^N - 1)
    # E[X] = 2^N * (4 - (N+2)*(1/2)^{N-1}) / (2^N - 1)
    def lrand(self):
        v = random.randint(1, self.rmax)
        return bin(v & -v).count('0')

    # 各ノード
    # node = [value, n_node0, ..., n_nodek, s_level]
    # n_nodei = レベルiの次のノード
    # 次のノードが終端の時は n_nodei = None

    # addition: O(logN)
    def add(self, x):
        s_level = self.lrand()
        level = self.level
        cur = self.root
        # 各レベルの追加元を検索
        prev = [None]*level
        for c_level in xrange(level-1, -1, -1):
            while cur[c_level+1] is not None and cur[c_level+1][0] < x:
                cur = cur[c_level+1]
            prev[c_level] = cur
        if cur[1] is not None and cur[1][0] == x:
            # avoid duplicate elements
            return

        new_node = [x] + [None]*level + [s_level]
        # 各レベルのノードに追加
        for c_level in xrange(s_level):
            if prev[c_level][c_level+1] is None:
                prev[c_level][c_level+1] = new_node
            else:
                new_node[c_level+1] = prev[c_level][c_level+1]
                prev[c_level][c_level+1] = new_node
        self.size += 1

    # deletion: O(logN)
    def delete(self, x):
        cur = self.root
        level = self.level
        prev = [None]*level
        # 各レベルで、削除ノードを指しているモノを検出
        for c_level in xrange(level-1, -1, -1):
            while cur[c_level+1] is not None and cur[c_level+1][0] < x:
                cur = cur[c_level+1]
            prev[c_level] = cur
        target = cur[1]
        if target is None or target[0] != x:
            return
        for c_level in xrange(level):
            # 同じオブジェクトをリンクから削除
            if prev[c_level] is not None and prev[c_level][c_level+1] is target:
                prev[c_level][c_level+1] = target[c_level+1]
        del target
        self.size -= 1

    # search k-th element: O(N)
    # k: 0-based
    def get(self, k):
        if not 0 <= k < self.size:
            return None
        cur = self.root
        for i in xrange(k+1):
            cur = cur[1]
        return cur[0]

    def visualize(self, padding=4):
        block = [[] for i in xrange(self.level+1)]
        cur = self.root
        block[0].append('r'.ljust(padding))
        for c_level in xrange(self.level):
            if cur[c_level+1] is None:
                block[c_level+1].append('-'.ljust(padding))
            else:
                block[c_level+1].append(str(cur[c_level+1][0]).ljust(padding))
        while cur[1] is not None:
            cur = cur[1]
            block[0].append(str(cur[0]).ljust(padding))
            s_level = cur[-1]
            for c_level in xrange(self.level):
                if cur[c_level+1] is None:
                    if c_level < s_level:
                        block[c_level+1].append('N'.ljust(padding))
                    else:
                        block[c_level+1].append('-'.ljust(padding))
                else:
                    assert c_level < s_level
                    block[c_level+1].append(str(cur[c_level+1][0]).ljust(padding))
        for e in block:
            print(''.join(e))

# SkipList (インデックスありでk番目要素アクセスにlogオーダー)
class IndexableSkipList:
    def __init__(self, level):
        self.level = level
        self.rmax = 2**level-1
        self.root = [None] * (level+2)
        self.size = 0

    # 1 ≦ k ≦ level
    # Pr(X = k) = 2^{k-1}/(2^N - 1)
    # E[X] = 2^N * (4 - (N+2)*(1/2)^{N-1}) / (2^N - 1)
    def lrand(self):
        v = random.randint(1, self.rmax)
        return bin(v & -v).count('0')

    # 各ノード
    # node = [value, [n_node0, wid0], ..., [n_nodek, widk], s_level]
    # n_nodei = レベルiの次のノード
    # widi    = レベルiの次のノードまでの幅
    # 次のノードが終端の時は [n_nodei, widi] = None

    # addition: O(logN)
    def add(self, x):
        s_level = self.lrand()
        level = self.level
        cur = self.root
        k = 0 # k: 追加位置-1
        # 各レベルの追加元を、位置を計算しながら検索
        prev = [None]*level
        for c_level in xrange(level-1, -1, -1):
            while cur[c_level+1] is not None and cur[c_level+1][0][0] < x:
                nxt, dk = cur[c_level+1]
                k += dk
                cur = nxt
            prev[c_level] = [cur, k]
        if cur[1] is not None and cur[1][0][0] == x:
            # avoid duplicate elements
            return

        # Y = new_node
        new_node = [x] + [None]*level + [s_level]

        # 各レベルのノードに追加
        for c_level in xrange(s_level):
            # X = cur
            cur, pos = prev[c_level]
            # X -> Z -> ...
            if cur[c_level+1] is None:
                # Z = NIL => X -> Y -> NIL
                cur[c_level+1] = [new_node, k + 1 - pos]
            else:
                # X -> Y -> Z -> ...
                # d(Y, Z) = pos(Z) - pos(Y) = pos(X) + d(X, Z) - pos(Y)
                new_node[c_level+1] = [cur[c_level+1][0], pos + cur[c_level+1][1] - k]
                # d(X, Y) = pos(Y) + 1 - pos(X)
                cur[c_level+1] = [new_node, k + 1 - pos]
        # 追加レベルより大きいノード間の幅を1大きくする
        for c_level in xrange(s_level, level):
            # X = cur, X -> Z -> ...
            cur, pos = prev[c_level]
            if cur[c_level+1] is not None:
                # d(X, Z) = d(X, Z) + 1
                cur[c_level+1][1] += 1
        self.size += 1

    # deletion: O(logN)
    def delete(self, x):
        cur = self.root
        level = self.level
        k = 0
        prev = [None]*level
        # 各レベルで、削除ノードを参照元を検出
        for c_level in xrange(level-1, -1, -1):
            while cur[c_level+1] is not None and cur[c_level+1][0][0] < x:
                nxt, dk = cur[c_level+1]
                k += dk
                cur = nxt
            prev[c_level] = [cur, k]
        # Y = target
        target = cur[1][0]
        if target is None or target[0] != x:
            return
        for c_level in xrange(level):
            # 同じオブジェクトをリンクから削除
            # X = cur
            cur, pos = prev[c_level]
            # X -> Z -> ...
            if cur[c_level+1] is not None:
                # Z != NIL
                if cur[c_level+1][0] is target:
                    # Z = Y
                    if target[c_level+1] is None:
                        # Y -> NIL => X -> NIL
                        cur[c_level+1] = None
                    else:
                        # d(X, Z) = d(X, Y) + d(Y, Z) - 1
                        cur[c_level+1] = [target[c_level+1][0], cur[c_level+1][1] + target[c_level+1][1] - 1]
                else:
                    # 削除レベルより大きいノード間の幅を1小さくする
                    # d(X, Z) = d(X, Z) - 1
                    cur[c_level+1][1] -= 1
        del target
        self.size -= 1

    # search k-th element: O(logN)
    # k: 0-based
    def get(self, k):
        if not 0 <= k < self.size:
            return None
        level = self.level
        cur = self.root
        pos = 0
        for c_level in xrange(level-1, -1, -1):
            while cur[c_level+1] is not None and pos + cur[c_level+1][1] < k+2:
                nxt, dpos = cur[c_level+1]
                pos += dpos
                cur = nxt
        return cur[0]

    def visualize(self, padding=7):
        block = [[] for i in xrange(self.level+1)]
        cur = self.root
        block[0].append('r'.ljust(padding))
        for c_level in xrange(self.level):
            if cur[c_level+1] is None:
                block[c_level+1].append('-'.ljust(padding))
            else:
                nxt, k = cur[c_level+1]
                block[c_level+1].append(("%d(%d)" % (nxt[0], k)).ljust(padding))
        while cur[1] is not None:
            cur = cur[1][0]
            block[0].append(str(cur[0]).ljust(padding))
            s_level = cur[-1]
            for c_level in xrange(self.level):
                if cur[c_level+1] is None:
                    if c_level < s_level:
                        block[c_level+1].append('N'.ljust(padding))
                    else:
                        block[c_level+1].append('-'.ljust(padding))
                else:
                    assert c_level < s_level
                    nxt, k = cur[c_level+1]
                    block[c_level+1].append(("%d(%d)" % (nxt[0], k)).ljust(padding))
        for e in block:
            print(''.join(e))

if __name__ == '__main__':
    print "SkipList"
    skiplist = SkipList(5)
    for i in xrange(10):
        skiplist.add(random.randint(1, 10**3-1))
    skiplist.visualize()
    for i in xrange(0, 10, 2):
        print i, skiplist.get(i)

    print "IndexableSkipList"
    skiplist = IndexableSkipList(5)
    for i in xrange(10):
        skiplist.add(random.randint(1, 10**3-1))
    skiplist.visualize()
    for i in xrange(0, 10, 2):
        print i, skiplist.get(i)

