# encoding: utf-8

# Floyd's cycle-finding algorithm

def cycle_detection(f, x0):
    p = q = x0

    # p = a_m, q = a_2m となるmを見つける
    # この時 m = kμ (μは循環部分の長さ) であり
    # a_{λ+i} = a_{λ+i+kμ} (λは非循環部分の長さ)
    while 1:
        p = f(p); q = f(f(q))
        if p == q:
            break

    # p = a_m, q = a_0 から p = q となるまで進める
    # a_m は循環部分に入って(m-λ)step進んでいるため
    #   - p = a_{λ+(m-λ)}  =(λstep)=>  a_{λ+m} = a_λ
    #   - q = a_0  =(λstep)=>  a_λ
    # よって p = q を求めることで λ が求まる
    p = x0; l = 0
    while p != q:
        l += 1
        p = f(p); q = f(q)

    # 最後にa_{λ+μ} = a_λ となるμを求める
    m = 0
    while 1:
        m += 1
        p = f(p)
        if p == q: break
    return l, m

print cycle_detection(lambda x: (x**2 + 2*x + 1) % (10**8+7), 7)
