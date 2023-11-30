#%% INTRODUÇÃO
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:29:18 2022

@author: Thaís Nakamura - RA: 148151 - Turma: I

Projeto 02 - Segmentação Média-Adventícia

Imagens Biomédicas - Matheus Cardoso

Motivação: Um Instituto renomado no Brasil faz todos os tipos de investigações,
intervenções, tratamentos e pesquisas relacionados ao coração. Dentre eles, estão
as investigações das coronárias, estas investigações são realizadas com um
equipamento chamado Ultrassom Intravascular (IVUS). Este equipamento adquire
imagens de seções transversais da coronária pela inserção e movimento de
retirada de um cateter. Contudo, além de não fornecer informações objetivas,
como dimensões, este equipamento fornece centenas de imagens de uma mesma seção,
o que dificulta muito a análise por segmentação manual. O pesquisador de cardiologia,
quem usa este equipamento para orientar doutorandos em medicina, pediu a criação
de um aplicativo para esta tarefa. O objetivo é fazer a segmentação e cálculo da área
da borda média-adventícia (parede externa do vaso) em imagens de IVUS.
"""

#%% INICIALIZAÇÃO
# importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import bibSeg
import bibButter

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
mask = bibButter.fazerMascaraButter2D(400, 400, 0.25, 4)
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