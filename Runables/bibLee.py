def leefilter(imagem, size, Cmin, Lmin, Cmax, Lmax, M, N):
    import scipy.signal
    import numpy as np
         
    # calculando média e variância
    varh = np.var(imagem[Lmin:Lmax,Cmin:Cmax])
    
    # criando a máscara e convoluindo
    w = np.ones((size,size), float)/(size*size)
    imagemmedia = scipy.signal.convolve2d(imagem, w, 'same')
    
    # varredura
    size = int(size)
    sizevarl = int((size-1)/2)
    varl = np.ones((M,N), float)
    for l in range(M-size):
        for c in range(N-size):
            varl[l+sizevarl,c+sizevarl] = np.var(imagem[(l):(l)+size, (c):(c)+size])
    
    # definindo k
    k = 1 - (varh/varl)
    k = np.clip(k, 0, 1)
    
    # substituindo o coeficiente K do filtro de Lee pelo gradiente
    wxs = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    wys = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    # gradiente
    dx = scipy.signal.convolve2d(imagem, wxs, 'same')
    dy = scipy.signal.convolve2d(imagem, wys, 'same')
    mod = np.power(np.power(dx,2)+np.power(dy,2),0.5)
    k = mod
    k = np.clip(k,0,1)
    
    # resultado
    imagemnew = imagemmedia + k*(imagem-imagemmedia)
    
    return(imagemnew)