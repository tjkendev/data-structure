class SkewHeap:
    def __init__(self):
        self.root = None

    @staticmethod
    def merge(p, q):
        if p is None:
            return q
        if q is None:
            return p

        if not p[2] < q[2]:
            p, q = q, p

        p[0], p[1] = p[1], p[0]
        p[0] = SkewHeap.merge(p[0], q)

        return p

    def push(self, v):
        # node: [left, right, key]
        new_node = [None, None, v]
        self.root = SkewHeap.merge(self.root, new_node)

    def pop(self):
        assert self.root is not None

        p_node = self.root

        self.root = SkewHeap.merge(p_node[0], p_node[1])
        return p_node[2]
