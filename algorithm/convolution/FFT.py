# encoding: utf-8

import cmath

# ======================================================
# 通常のDFT O(N^2)
def dft(f):
    exp = cmath.exp
    pi = cmath.pi
    N = len(f)
    R = range(N)
    F = [sum(exp(-2j*k*n*pi/N)*f[n] for n in R) / N for k in R]
    return F
def rdft(F):
    exp = cmath.exp
    pi = cmath.pi
    N = len(F)
    R = range(N)
    f = [sum(exp(2j*k*n*pi/N)*F[k] for k in R) for n in R]
    return f

# ======================================================
# Cooley-Tukeyアルゴリズム

# FFT O(Nlog(N))
# 1つ飛ばしの配列生成に時間かかって遅いっぽい
def fft(f):
    exp = cmath.exp
    pi = cmath.pi
    def fft_dfs(f, k, w, wk):
        N = len(f)
        if N==1: return [f[0], f[0]]
        w2 = w**2
        wk2 = wk**2
        U = fft_dfs(f[0:N:2], k, w2, wk2)
        V = fft_dfs(f[1:N:2], k, w2, wk2)
        return [U[0] + V[0]*wk, U[0] - V[0]*wk]
    N = len(f)
    F = [0.0] * N
    w = exp(-2j*pi/N)
    for k in xrange(N/2):
        F[k], F[k+N/2] = fft_dfs(f, k, w, w**k)
    for k in xrange(N):
        F[k] /= N
    return F

def rfft(F):
    exp = cmath.exp
    pi = cmath.pi
    def rfft_dfs(F, n, w, wn):
        N = len(F)
        if N==1: return [F[0], F[0]]
        w2 = w**2
        wn2 = wn**2
        U = rfft_dfs(F[0:N:2], n, w2, wn2)
        V = rfft_dfs(F[1:N:2], n, w2, wn2)
        return [U[0] + V[0]*wn, U[0] - V[0]*wn]
    N = len(F)
    f = [0.0] * N
    w = exp(2j*pi/N)
    for n in xrange(N/2):
        f[n], f[n+N/2] = rfft_dfs(F, n, w, w**n)
    return f

# ======================================================
# bit反転ありFFT
# 1つ飛びの配列分割をしないように
# 最初から値を並べておく
# 早い
# n0以上のn=2^mについてビット反転のインデックスを返す
br_dic = {}
def bit_reversal(n0):
    if n0 in br_dic: return br_dic[n0]
    n = 1
    while n<n0: n <<= 1
    if n in br_dic:
        br_dic[n0] = d = filter(lambda x: x<n0, br_dic[n])
        return d
    d = range(n)
    ns = n>>1
    ns1 = ns + 1
    i = 0
    for j in xrange(0, ns, 2):
        if j<i:
            d[i], d[j] = d[j], d[i]
            d[i+ns1], d[j+ns1] = d[j+ns1], d[i+ns1]
        d[i+1], d[j+ns] = d[j+ns], d[i+1]
        k = ns >> 1; i ^= k
        while k > i:
            k >>= 1; i ^= k
    br_dic[n] = d
    br_dic[n0] = d = filter(lambda x: x<n0, d)
    return d

# 配列サイズを2^Nに合わせるようにゼロ埋め
def zero_fill_2N(d):
    N = len(d)
    N2 = 2**(N-1).bit_length()
    return d + [0]*(N2-N)

# サイズを2^Nに合わせ、かつbit反転する
def bit_reversal_2N(d):
    n0 = len(d)
    # X&(X-1)==0 --> X = 2^M
    d = zero_fill_2N(d) if n0&(n0-1) else d[:]
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

def br_fft(f):
    def fft_dfs(f, l, N, wk):
        if N==1: return f[l]
        if N==2: return f[l] + f[l+1]*wk
        wk2 = wk**2; dh = dhs[N]
        return fft_dfs(f, l, dh, wk2) + fft_dfs(f, l+dh, N-dh, wk2)*wk
    N = len(f)
    dhs = [(n+1)/2 for n in xrange(N+1)]
    br = bit_reversal(N)
    fb = [f[br[i]]/N for i in xrange(N)]
    F = [0] * N
    w = cmath.exp(-2j*cmath.pi/N); wk = 1+0j
    if N%2:
        for k in xrange(N):
            F[k] = fft_dfs(fb, 0, N, wk)
            wk *= w
    else:
        d2 = (N+1)/2; dN = N-d2
        for k in xrange(N/2):
            F[k] = F[k+N/2] = fft_dfs(fb, 0, d2, wk**2)
            V = fft_dfs(fb, d2, dN, wk**2)*wk
            F[k] += V; F[k+N/2] -= V
            wk *= w
    return F

def br_rfft(F):
    def rfft_dfs(F, l, N, wn):
        if N==1: return F[l]
        if N==2: return F[l] + F[l+1]*wn
        wn2 = wn**2; d2 = ds[N]
        return rfft_dfs(F, l, d2, wn2) + rfft_dfs(F, l+d2, N-d2, wn2)*wn
    N = len(F)
    ds = [(i+1)/2 for i in xrange(N+1)]
    br = bit_reversal(N)
    Fb = [F[br[i]] for i in xrange(N)]
    f = [0] * N
    w = cmath.exp(2j*cmath.pi/N); wn = 1+0j
    if N%2:
        for n in xrange(N):
            f[n] = rfft_dfs(Fb, 0, N, wn)
            wn *= w
    else:
        d2 = (N+1)/2; dN = N-d2
        for n in xrange(N/2):
            f[n] = f[n+N/2] = rfft_dfs(Fb, 0, d2, wn**2)
            V = rfft_dfs(Fb, d2, dN, wn**2)*wn
            f[n] += V; f[n+N/2] -= V
            wn *= w
    return f

# ======================================================
# 一つの再帰で1~Nの計算を行うようにしたFFT
# 普通に速かった...
def maxN2(N):
    N0 = 1
    while N0<N: N0 <<= 1
    return N0

exp_table = {}
def exp_table_init(N):
    exp = cmath.exp
    exp_table[0] = 1.0
    if N<0:
        base = -2j*cmath.pi
        N = -N
        while N:
            exp_table[-N] = exp(base/N)
            N >>= 1
    else:
        base = 2j*cmath.pi
        while N:
            exp_table[N] = exp(base/N)
            N >>= 1

def br_fft2(f):
    fb = bit_reversal_2N(f)
    N = len(fb)
    if N==1: return fb
    for i in xrange(N): fb[i] /= N
    exp_table_init(-N)
    def fft_dfs(f, s, N):
        if N==2: return [f[s]+f[s+1], f[s]-f[s+1]]
        N2 = N/2
        F0 = fft_dfs(f, s, N2)
        F1 = fft_dfs(f, s+N2, N2)
        F = [0] * N
        w = exp_table[-N]; wk = 1.0
        for k in xrange(N2):
            F[k] = F0[k] + wk * F1[k]
            F[k+N2] = F0[k] - wk * F1[k]
            wk *= w
        return F
    return fft_dfs(fb, 0, N)

def br_rfft2(F):
    Fb = bit_reversal_2N(F)
    N = len(Fb)
    if N==1: return Fb
    exp_table_init(N)
    def rfft_dfs(F, s, N):
        if N==2: return [F[s]+F[s+1], F[s]-F[s+1]]
        N2 = N/2
        f0 = rfft_dfs(F, s, N2)
        f1 = rfft_dfs(F, s+N2, N2)
        f = [0] * N
        w = exp_table[N]; wn = 1.0
        for n in xrange(N2):
            f[n] = f0[n] + wn * f1[n]
            f[n+N2] = f0[n] - wn * f1[n]
            wn *= w
        return f
    return rfft_dfs(Fb, 0, N)

# ------------------------------------------------------
# 別にbit-reversalしなくても、位置と間隔さえ覚えていれば参照できることに気づいた
# Stockhamアルゴリズムっぽいけど、処理内容はCooley-Tukeyアルゴリズムっぽい
# Pythonは要素数が少ないほどアクセスよさそうな気がする
def d_fft(f):
    f = zero_fill_2N(f)
    N = len(f)
    if N==1: return f
    for i in xrange(N): f[i] /= N
    exp_table_init(-N)
    exp_t = exp_table
    def fft_dfs(f, s, N, st):
        if N==2:
            a = f[s]; b = f[s+st]
            return [a+b, a-b]
        N2 = N/2; st2 = st*2
        F0 = fft_dfs(f, s   , N2, st2)
        F1 = fft_dfs(f, s+st, N2, st2)
        w = exp_t[-N]; wk = 1.0
        for k in xrange(N2):
            U = F0[k]; V = wk * F1[k]
            F0[k] = U + V
            F1[k] = U - V
            wk *= w
        F0.extend(F1)
        return F0
    return fft_dfs(f, 0, N, 1)

def d_rfft(F):
    F = zero_fill_2N(F)
    N = len(F)
    if N==1: return F
    exp_table_init(N)
    exp_t = exp_table
    def rfft_dfs(F, s, N, st):
        if N==2:
            A = F[s]; B = F[s+st]
            return [A+B, A-B]
        N2 = N/2; st2 = st*2
        f0 = rfft_dfs(F, s   , N2, st2)
        f1 = rfft_dfs(F, s+st, N2, st2)
        w = exp_t[N]; wn = 1.0
        for n in xrange(N2):
            U = f0[n]; V = wn * f1[n]
            f0[n] = U + V
            f1[n] = U - V
            wn *= w
        f0.extend(f1)
        return f0
    return rfft_dfs(F, 0, N, 1)

# ------------------------------------------------------
# fft_dfsとrfft_dfsの処理がだいたい一緒だったので外に出した
# たぶんこれがシンプルになってよさげ？
def fft_dfs_out(f, s, N, st, exp_t):
    if N==2:
        a = f[s]; b = f[s+st]
        return [a+b, a-b]
    N2 = N/2; st2 = st*2
    F0 = fft_dfs_out(f, s   , N2, st2, exp_t)
    F1 = fft_dfs_out(f, s+st, N2, st2, exp_t)
    w = exp_t[N]; wk = 1.0
    for k in xrange(N2):
        U = F0[k]; V = wk * F1[k]
        F0[k] = U + V
        F1[k] = U - V
        wk *= w
    F0 += F1
    return F0

def make_exp_t(N, base):
    exp = cmath.exp
    exp_t = {0: 1}
    temp = N
    while temp:
        exp_t[temp] = exp(base / temp)
        temp >>= 1
    return exp_t

def d_fft_out(f):
    f = f[:]
    #f = zero_fill_2N(f)
    N = len(f)
    if N==1: return f
    for i in xrange(N): f[i] /= N
    exp_t = make_exp_t(N, -2j*cmath.pi)
    return fft_dfs_out(f, 0, N, 1, exp_t)

def d_rfft_out(F):
    #F = zero_fill_2N(F)
    N = len(F)
    if N==1: return F
    exp_t = make_exp_t(N, 2j*cmath.pi)
    return fft_dfs_out(F, 0, N, 1, exp_t)

# ======================================================
# Split-Radixアルゴリズム

# TODO

# ======================================================
# Stockhamアルゴリズム

# こちらを参考にStockhamアルゴリズムを実装
# http://members3.jcom.home.ne.jp/zakii/fourie/32_implement_software.htm
# なんか微妙に遅い...
def d_fft2(f):
    f = zero_fill_2N(f)
    N = len(f)
    #if N==1: return f
    for i in xrange(N): f[i] /= N

    exp = cmath.exp; base = -2j*cmath.pi/N
    N2 = N/2
    s = 1; l = N2
    F = f; F1 = [0]*N
    while l: # s < N
        w = exp(base * l)
        for i in xrange(l):
            wk = 1
            si = i*s; sis = si + s
            for k in xrange(si, si+s):
                U = F[k]; V = F[k+N2] * wk
                F1[k+si]  = U + V
                F1[k+sis] = U - V
                wk *= w
        F, F1 = F1, F
        s *= 2; l /= 2
    return F

def d_rfft2(F):
    F = zero_fill_2N(F)
    N = len(F)
    #if N==1: return F

    exp = cmath.exp; base = 2j*cmath.pi/N
    N2 = N/2
    s = 1; l = N2
    f = F; f1 = [0]*N
    while l: # s < N
        w = exp(base * l)
        for i in xrange(l):
            wk = 1
            si = i*s; sis = si + s
            for k in xrange(si, si+s):
                U = f[k]; V = f[k+N2] * wk
                f1[k+si]  = U + V
                f1[k+sis] = U - V
                wk *= w
        f, f1 = f1, f
        s *= 2; l /= 2
    return f
