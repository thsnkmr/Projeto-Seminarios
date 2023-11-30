def fazerMascaraGaussiana2D(M, N, fc):
    import numpy as np
    H_Gauss = np.zeros((M, N), complex)
    Do = fc * (M/2)
    
    for l in range(M):
        for c in range(N):
            distx = c - (N/2)
            disty = l - (M/2)
            dist = np.math.sqrt(distx**2 + disty**2)
            H_Gauss[l,c] = np.exp(-1*(dist**2)/(2*(Do**2)))
    return H_Gauss