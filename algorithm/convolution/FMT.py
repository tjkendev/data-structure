# encoding: utf-8

# FMTは、複素数で演算を行うFFTを整数環上で行うものである。
# ωとしてω^n ≡ 1 (mod P)となるω、P、nを用いる。
#
# ここでは、
# f(x) = a_{n-1}*x^{n-1} + ... + a_1*x + a_0とする。
#
# - 順変換
# 0 ≦ i ≦ n-1について、
# f_k = f(ω^k) (mod P) = \sum_{i=0}^{n-1} a_i*ω^{ik} (mod P)
# を計算
#
# - 逆変換
# 0 ≦ k ≦ n-1について
# a_i = \sum_{k=0}^{n-1} a_i*ω^{-ik} (mod P)
# を計算

# ========================================
# 計算に必要なパラメタ
# ω^n ≡ 1 (mod P) となるようなω, n, Pを選ぶ
# Pは素数, nは2^mが楽
omega = 55
n = 2**18
P = 1048*n + 1

rev = pow(omega, P-2, P)


# ========================================
# 愚直なO(N^2)の整数環DFT
# 畳み込む要素をL個に制限することでちょっと早くできる

def naive_dft(f, l=None):
    F = [0]*n
    l = l or n
    for i in xrange(n):
        base = pow(omega, i, P)
        cur = 1
        for j in xrange(l):
            F[i] = (F[i] + cur*f[j]) % P
            cur = (base * cur) % P
    return F

def naive_idft(F, l=None):
    f = [0]*n
    l = l or n
    for i in xrange(l):
        base = pow(rev, i, P)
        cur = 1
        for j in xrange(n):
            f[i] = (f[i] + cur*F[j]) % P
            cur = (base * cur) % P
        f[i] = (f[i] * pow(n, P-2, P)) % P
    return f


# ========================================
# O(NlogN)の整数FMT
# 再帰的に求める。畳み込み対象の数LIMで処理を少し早くしてる
def fmt_dfs(A, s, N, st, base, half, lim):
    if N == 2:
        a = A[s]; b = A[s+st]
        return [(a+b)%P, (a+b*base)%P]
    F = [0]*N
    if s < lim:
        N2 = N>>1; st2 = st<<1; base2 = pow(base, 2, P)
        F0 = fmt_dfs(A, s   , N2, st2, base2, half, lim)
        F1 = fmt_dfs(A, s+st, N2, st2, base2, half, lim)
        wk = 1
        for k in xrange(N2):
            U = F0[k]; V = F1[k] * wk
            F[k] = (U + V) % P
            F[k+N2] = (U + V*half) % P
            wk = (wk * base) % P
    return F

def fmt(f, l):
    if l == 1:
        return f
    return fmt_dfs(f, 0, n, 1, omega, pow(omega, n/2, P), l)

def ifmt(F, l):
    if l == 1:
        return F
    f = fmt_dfs(F, 0, n, 1, rev, pow(rev, n/2, P), n)
    n_rev = pow(n, P-2, P)
    return [(e * n_rev) % P for e in f]


# ========================================
# O(NlogN)の整数FMT
# bit反転を利用して、ボトムアップにループでFMTを行う
# ATC001 - C問題: 高速フーリエ変換 (AC): http://atc001.contest.atcoder.jp/submissions/1051678

# 配列要素のbit反転
def bit_reverse(d):
    # X&(X-1)==0 --> X = 2^M
    n = len(d)
    ns = n>>1; nss = ns>>1
    ns1 = ns + 1
    i = 0
    for j in xrange(0, ns, 2):
        if j<i:
            d[i], d[j] = d[j], d[i]
            d[i+ns1], d[j+ns1] = d[j+ns1], d[i+ns1]
        d[i+1], d[j+ns] = d[j+ns], d[i+1]
        k = nss; i ^= k
        while k > i:
            k >>= 1; i ^= k
    return d

# ボトムアップのFMTを行う
def fmt_bu(A, n, base, half, Q):
    N = n
    m = 1
    while n>1:
        n >>= 1
        w = pow(base, n, Q)
        wk = 1
        for j in xrange(m):
            for i in xrange(j, N, 2*m):
                U = A[i]; V = (A[i+m]*wk) % Q
                A[i] = (U + V) % Q
                A[i+m] = (U + V*half) % Q
            wk = (wk * w) % Q
        m <<= 1
    return A

def fmt(f, l, Q=P):
    if l == 1: return f
    A = f[:]
    bit_reverse(A)
    return fmt_bu(A, n, omega, pow(omega, n/2, Q), Q)

def ifmt(F, l, Q=P):
    if l == 1: return F
    A = F[:]
    bit_reverse(A)
    f = fmt_bu(A, n, rev, pow(rev, n/2, Q), Q)
    n_rev = pow(n, Q-2, Q)
    return [(e * n_rev) % Q for e in f]

def convolute(a, b, l, Q=P):
    A = fmt(a, l, Q)
    B = fmt(b, l, Q)
    C = [(s * t) % Q for s, t in zip(A, B)]
    c = ifmt(C, l, Q)
    return c
