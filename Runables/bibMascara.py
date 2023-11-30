def fazerMascaraGauss2D(media, desvio):
    import scipy.signal
    import numpy as np
    
    tamanho = 2*media + 1
    
    g = scipy.signal.gaussian(tamanho, std=desvio)
    g1 = np.zeros([tamanho,tamanho])
    g1[media,:] = g
    gT1 = np.transpose(g1)
    wGauss2D = scipy.signal.convolve2d(g1, gT1, 'same')
    wGauss2DNorm = wGauss2D/(np.sum(wGauss2D))
        
    return wGauss2DNorm