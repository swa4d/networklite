import numpy as np
import math

a = 0.8406
b = 7.522e-5
R = 8.314
T = 288.15

def get_roots(P):
    coeffs = [1, -(b + R*T/P), a/P, -(a*b/P)]
    return np.roots(coeffs)

def get_Z(P, V):
    return (P * V) / (R * T)

def get_f_vdw(Z, V, P):
    exponent = Z - 1 - math.log(Z) + math.log(V/(V-b)) - a/(V*R*T)
    return P * math.exp(exponent)

def get_df_from_roots(roots, P):
    real_roots = sorted(
        [r.real for r in roots],
        reverse=True
    )
    if len(real_roots) < 2:
        return None

    vV, vL = real_roots[0], real_roots[-1]
    fV = get_f_vdw(get_Z(P, vV), vV, P)
    fL = get_f_vdw(get_Z(P, vL), vL, P)
    return fV - fL

def main():
    prev_delta = None
    for i in range(100, 5500, 10):
        P = i * 1000
        delta = get_df_from_roots(get_roots(P), P)
        if delta is None:
            continue
        if prev_delta is not None and prev_delta * delta < 0:
            print(f"Psat between {(i-10)/1e3:.2f} and {i/1e3:.2f} MPa")
        print(f"P = {P/1e6:.3f} MPa    deltaF = {delta:+.4f}")
        prev_delta = delta

if __name__ == "__main__":
    main()