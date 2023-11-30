def fazerMascaraButter2D (M, N, fc, n):
    import numpy as np
    H_Butter = np.zeros((M, N), complex)
    Do = fc * (M/2)
    
    for l in range(M):
        for c in range(N):
            distx = c - (N/2)
            disty = l - (M/2)
            dist = np.math.sqrt(distx**2 + disty**2)
            H_Butter[l,c] = 1/(1 + (dist/Do)**(2*n))

    return H_Butter