class LeftistHeap:
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

        p[1] = LeftistHeap.merge(p[1], q)
        if p[0] is None:
            p[0], p[1] = p[1], p[0]
            p[3] = 0
            return p
        if p[0][3] < p[1][3]:
            p[0], p[1] = p[1], p[0]
        p[3] = p[1][3] + 1
        return p

    def push(self, v):
        # node: [left, right, key, npl]
        new_node = [None, None, v, 0]
        self.root = LeftistHeap.merge(self.root, new_node)

    def pop(self):
        assert self.root is not None

        p_node = self.root

        self.root = LeftistHeap.merge(p_node[0], p_node[1])
        return p_node[2]
