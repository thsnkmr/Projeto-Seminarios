#%% INTRODUÇÃO
# -*- coding: utf-8 -*-

#%% INICIALIZAÇÃO
# importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import bibSeg
import bibFuncaoTransferencia

# leitura e normalização da imagem original
f = cv2.imread('img19.pgm', 0)
f = skimage.img_as_float(f)
(M, N) = np.shape(f)
roi = cv2.selectROI(f)
Cmin = roi[0]
Lmin = roi[1]
Cmax = roi[0] + roi[2]
Lmax = roi[1] + roi[3]

plt.figure()
plt.title('IVUS')
plt.imshow(f, cmap='gray')

# FFT para processamento na frequência
ff = np.fft.fft2(f)
ff = np.fft.fftshift(ff)

# leitura e normalização do gold standard
GS = cv2.imread('gsmab0.pgm', 0)
GS = skimage.img_as_float(GS)
thresh = 0.5
GS = GS > thresh

plt.figure()
plt.title('GS')
plt.imshow(GS, cmap='gray')

#%% PRÉ-PROCESSAMENTO
# filtrando a imagem no domínio da frequência
mask = bibFuncaoTransferencia.fazerMascaraIdeal2D(400, 400, 0.25)
filtragem = ff * mask

# realizando a IFFT
filtered = np.fft.ifft2(filtragem)
filteredreal = np.abs(filtered)

# plotando a imagem filtrada
plt.figure()
plt.title('Imagem filtrada')
plt.imshow(filteredreal, cmap='gray')

# alongando o contraste
flong = skimage.exposure.rescale_intensity(filteredreal, in_range=(0.0335, 0.6045))

# plotando a imagem
plt.figure()
plt.title('Imagem com alongamento de contraste')
plt.imshow(flong, cmap='gray')

#%% SEGMENTAÇÃO
# média e desvio padrão da região de interesse
m = np.mean(flong[Lmin:Lmax, Cmin:Cmax])
d = np.std(flong[Lmin:Lmax, Cmin:Cmax])
      
ObjSeg = np.zeros((M, N), float)
for l in range(M):
    for c in range(N):
    	if ((flong[l,c] > (m-(3*d))) & (flong[l,c] < (m+(3*d)))):
    		ObjSeg[l,c] = flong[l,c]
        
ObjSegBin = ObjSeg > 0.0335

# correção da câmera
for l in range(M):
    for c in range(N):
       distx = c - 200
       disty = l - 200
       dist = np.math.sqrt(distx**2 + disty**2)
       if dist <= 38:
           ObjSegBin[l,c] = 1

# plotando a imagem
plt.figure()
plt.title('Objeto segmentado original')
plt.imshow(ObjSegBin, cmap='gray')

# mostrando resultados
resultado = bibSeg.fazerAva(ObjSegBin, GS)   
print('resultado segmentação: ', resultado)
