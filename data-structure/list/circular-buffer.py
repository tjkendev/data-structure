# encoding: utf-8

# Circular Buffer (Circular Queue)
# 固定長の配列を循環するようにQueueとして扱う

class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.data = [None]*size
        self.top = 0
        self.bottom = 0

    def rpush(self, x):
        assert self.top - self.bottom < self.size
        self.top += 1
        self.data[self.top % self.size] = x

    def rpop(self):
        if self.bottom == self.top:
            return None
        self.top -= 1
        return self.data[(self.top+1) % self.size]

    def lpush(self, x):
        assert self.top - self.bottom < self.size
        self.data[self.bottom % self.size] = x
        self.bottom -= 1

    def lpop(self):
        if self.bottom == self.top:
            return None
        self.bottom += 1
        return self.data[self.bottom % self.size]

    def __len__(self):
        return self.top - self.bottom

if __name__ == '__main__':
    buf = CircularBuffer(10)
    for i in xrange(10):
        buf.rpush(i)
    for i in xrange(11):
        print buf.lpop()
