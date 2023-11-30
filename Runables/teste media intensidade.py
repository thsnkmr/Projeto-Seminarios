import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import cv2
import skimage

# lendo a imagem
bebe = cv2.imread('img19.pgm',0)
bebe = skimage.img_as_float(bebe)
plt.figure()
plt.title('bebe')
plt.imshow(bebe, cmap = 'gray')

# definindo os parâmetros
roi = cv2.selectROI(bebe)
Cmin = roi[0]
Lmin = roi[1]
Cmax = roi[0] + roi[2]
Lmax = roi[1] + roi[3]

# calculando média e variância
media = np.mean(bebe[Lmin:Lmax,Cmin:Cmax])
varh = np.var(bebe[Lmin:Lmax,Cmin:Cmax])

# criando a máscara e convoluindo
w = np.ones((7,7), float)/(7*7) # máscara filtro média
bebemedia = scipy.signal.convolve2d(bebe,w,'same')

# varredura
(M,N) = np.shape(bebe)
varl = np.ones((M,N), float)
for l in range(M-7):
    for c in range(N-7):
        varl[l+3,c+3] = np.var(bebe[l:l+7,c:c+7])

# definindo k
k = 1 - (varh/varl)
k = np.clip(k,0,1)

# resultado
bebelee = bebemedia + k*(bebe-bebemedia)

# plotando
plt.figure()
plt.title('bebe media')
plt.imshow(bebemedia, cmap='gray')

plt.figure()
plt.title('bebe lee')
plt.imshow(bebelee, cmap='gray')

# DESAFIO: substituindo o coeficiente K do filtro de Lee pelo gradiente
wxs = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
wys = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

# gradiente
dx = scipy.signal.convolve2d(bebe, wxs, 'same')
dy = scipy.signal.convolve2d(bebe, wys, 'same')
mod = np.power(np.power(dx,2)+np.power(dy,2),0.5)
k = mod
k = np.clip(k,0,1)

# resultado
bebeleenew = bebemedia + k*(bebe-bebemedia)

# plotando
plt.figure()
plt.title('bebe lee new')
plt.imshow(bebeleenew, cmap='gray')