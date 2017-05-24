# encoding: utf-8
import random, itertools

random.seed()
randint = random.randint
choice = random.choice
shuffle = random.shuffle

# n変数によるm個のclauseから成る論理式Fを生成
def make(n, m):
    # res: m個のclauseのAND結合
    res = []
    for i in xrange(m):
        temp = [-n]
        while sum(temp) == -n:
            # temp: 一つのclauseのOR結合
            # temp[i] => 1: Xi, 0: なし, -1: ~Xi
            # 例) [1, 0, -1] => (X0+~X2)
            temp = [randint(-1, 1) for i in xrange(n)]
        res.append(temp)
    return res

# 単純な総当り (2^n)
def bruteforce_solver(n, m, F):
    for X in itertools.product([0, 1], repeat=n):
        for C in F:
            for i in xrange(n):
                if C[i] == X[i]: break
            else: break
        else: return X
    return None

# MS(Monien-Speckenmeyer) Algorithm
# k=3 => O(1.8393^n)
def ms_solver(n, m, F):
    X = [-1]*n
    def dfs(i):
        if i == m: return True
        C = F[i]
        ok = 1
        P = [0]*n
        for j in xrange(n):
            if C[j]!=-1 and X[j]==-1:
                P[j] = 1
                X[j] = C[j]
                if dfs(i+1):
                    return True
                ok = 0
                X[j] ^= 1
        if ok:
            return dfs(i+1)
        for j in xrange(n):
            if P[j]:
                X[j] = -1
        return False
    return X if dfs(0) else None

# MS(Monien-Speckenmeyer) Algorithm
# k=3 => O(1.6181^n)
# この実装はTODO

# Schoning's algorithm
#  (指数部分の)計算量 (2-2/k)^(n*t) [k=3 => (1.33333)^n]
#  t: 試行回数
def schoning_solver(n, m, F):
    G = [[i for i in xrange(n) if C[i]!=-1] for C in F]
    for k in xrange(10):
        X = [randint(0, 1) for i in xrange(n)]
        # 3n回のランダムウォークで十分
        for t in xrange(3*n):
            unsat = []
            # 充足チェック
            for i, C in enumerate(F):
                for j in xrange(n):
                    if C[j] == X[j]: break
                else:
                    unsat.append(i)
            # 全て充足している場合 => それを返す
            if not unsat: return X
            # 充足していないclauseの中の一つの変数の値を変更 (=> ランダムウォーク)
            X[choice(G[choice(unsat)])] ^= 1
    # 最後まで見つからなかった場合
    return None

# PPZ Algorithm
def ppz_solver(n, m, F):
    Xc = [[j for j, C in enumerate(F) if C[i]!=-1] for i in xrange(n)]
    for k in xrange(10):
        X = [0]*n
        sat = [0] * m
        Cc = [sum(1 for i in xrange(n) if C[i]!=-1) for C in F]
        idx = range(n)
        shuffle(idx)
        for i in idx:
            for j in Xc[i]:
                if not sat[j] and Cc[j] == 1:
                    X[i] = F[j][i]
                    break
            else:
                X[i] = randint(0, 1)

            for j in Xc[i]:
                if not sat[j] and F[j][i] == X[i]:
                    sat[j] = 1
                Cc[j] -= 1
        if sum(sat) == m: return X
    return None

# 論理式を表示
def print_formula(F):
    lst = []
    for C in F:
        lst.append("(" + "+".join(("X%d" if C[i] else "~X%d") % i for i in xrange(len(C)) if C[i]!=-1) + ")")
    print "*".join(lst)

# ===== example =====
n = 3
m = 5
F = make(n, m)
print_formula(F)
print schoning_solver(n, m, F)
