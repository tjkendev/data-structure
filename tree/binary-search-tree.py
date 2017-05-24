# encoding: utf-8

# node = [None, A, B]
class BinarySearchTree:
    """ ２分探索木 """

    root = None

    def find(self, x):
        """ 要素xが存在するかチェック """
        node = self.root
        if node is None:
            return None
        while 1:
            v, left, right, parent = node
            if x == v:
                return x
            elif x < v:
                if left is None:
                    return None
                node = left
            else:
                if right is None:
                    return None
                node = right

    def insert(self, x):
        """ 要素xを挿入 """
        node = self.root
        if node is None:
            self.root = [x, None, None, None]
            return
        while 1:
            v, left, right, parent = node
            if x < v:
                if left is None:
                    node[1] = [x, None, None, node]
                    return
                node = left
            else:
                if right is None:
                    node[2] = [x, None, None, node]
                    return
                node = right

    def delete(self, x):
        """ 要素xを削除 """
        node = self.root
        if node is None:
            return
        which = -1
        while 1:
            v, left, right, parent = node
            if v == x:
                # 要素が見つかった時
                if left is None and right is None:
                    # [None <- node -> None]
                    # nodeを削除するのみ
                    if which == -1:
                        self.root = None
                    else:
                        parent[which] = None
                elif left is None and right is not None:
                    # [None <- node -> B]
                    # Bをnodeの位置に持ってくる
                    if which == -1:
                        self.root = right
                        right[3] = self.root
                    else:
                        parent[which] = right
                        right[3] = parent
                elif left is not None and right is None:
                    # [A <- node -> None]
                    # Aをnodeの位置に持ってくる
                    if which == -1:
                        self.root = left
                        left[3] = self.root
                    else:
                        parent[which] = left
                        left[3] = parent
                else:
                    # [A <- node -> B]
                    # nodeを削除して、左nodeの内の最大の要素をnodeの位置に持ってくる
                    lnode = left
                    while lnode[2] is not None:
                        lnode = lnode[2]
                    if lnode is left:
                        # 最大の要素の親nodeが削除されるnodeの場合
                        if which == -1:
                            self.root = lnode
                        else:
                            parent[which] = lnode
                        lnode[2], lnode[3] = node[2], node[3]
                        if node[2] is not None:
                            node[2][3] = lnode
                    else:
                        # 最大の要素の親nodeが削除されるnodeでない場合
                        lnode[3][2] = lnode[1]
                        if lnode[1] is not None:
                            lnode[1][3] = lnode[3]

                        if which == -1:
                            self.root = lnode
                        else:
                            parent[which] = lnode
                        lnode[1], lnode[2], lnode[3] = node[1], node[2], node[3]
                        if node[1] is not None:
                            node[1][3] = lnode
                        if node[2] is not None:
                            node[2][3] = lnode
                return
            if x < v:
                if left is None:
                    return
                node = left
                which = 1
            else:
                if right is None:
                    return
                node = right
                which = 2

    def get(self, node=None):
        """ 木に含まれる(ソートされた)要素のリストを求める """
        node = node or self.root
        if node is None:
            return []
        v, left, right, parent = node
        lres = [] if left is None else self.get(left)
        rres = [] if right is None else self.get(right)
        return lres + [v] + rres



if __name__ == '__main__':
    import random
    random.seed()

    def check(node):
        if node is None:
            return
        v, left, right, parent = node
        if left is not None:
            assert left[3] is node
            check(left)
        if right is not None:
            assert right[3] is node
            check(right)

    tree = BinarySearchTree()
    for i in xrange(100):
        v = random.randint(1, 5)
        if random.randint(0, 1) > 0:
            tree.insert(v)
            print "add", v
        else:
            tree.delete(v)
            print "del", v
        check(tree.root)
        print tree.get()
