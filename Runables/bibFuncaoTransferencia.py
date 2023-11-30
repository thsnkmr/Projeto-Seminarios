def fazerMascaraIdeal2D(M, N, fc):
    import numpy as np
    H_Ideal = np.zeros((M, N), complex)
    Do = fc * (M/2)
    
    for l in range(M):
        for c in range(N):
            distx = c - (N/2)
            disty = l - (M/2)
            dist = np.math.sqrt(distx**2 + disty**2)
            if dist <= Do:
                H_Ideal[l,c] = 1 + 0j
    return H_Ideal

# cálculo de D e Do de acordo com a frequência de corte Do = fc.raiomax