# criando a função

def fazerHistograma(imagem, M, N):
    # importar bibliotecas
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 # OpenCV
    # declarar vetor histograma inicialmente tudo com 0
    histograma = np.zeros((256), int)
    
    for m in range(M):
        for n in range(N):
            histograma[imagem[m, n]] = histograma[imagem[m, n]] + 1    
                                                      
    return(histograma)