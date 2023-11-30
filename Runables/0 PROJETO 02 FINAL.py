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
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import bibGrow
import bibSeg

# leitura e normalização da imagem original
f = cv2.imread('img2.pgm', 0)
f = skimage.img_as_float(f)
(M, N) = np.shape(f)

# capturando região de interesse
roi = cv2.selectROI(f)
cv2.destroyAllWindows()
Cmin = roi[0]
Lmin = roi[1]
Cmax = roi[0] + roi[2]
Lmax = roi[1] + roi[3]

# mostrando a imagem original
plt.figure()
plt.title('IVUS')
plt.imshow(f, cmap='gray')

# leitura e normalização do gold standard
GS = cv2.imread('gsmab2.pgm', 0)
GS = skimage.img_as_float(GS)
thresh= 0.5
GS = GS > thresh

# mostrando o gold standard
plt.figure()
plt.title('GS')
plt.imshow(GS, cmap='gray')

#%% PRÉ-PROCESSAMENTO (FILTRAGEM E ALONGAMENTO DE CONTRASTE)
# filtrando a imagem com filtro média 5x5
filteredreal = scipy.signal.medfilt2d(f, kernel_size=5)

# plotando a imagem filtrada
plt.figure()
plt.title('Imagem filtrada')
plt.imshow(filteredreal, cmap='gray')

# alongando o contraste
flong = skimage.exposure.rescale_intensity(filteredreal, in_range=(0.0324, 0.5998))

# plotando a imagem realçaçada
plt.figure()
plt.title('Imagem com alongamento de contraste')
plt.imshow(flong, cmap='gray')

#%% SEGMENTAÇÃO (CRESCIMENTO DE REGIÃO SEMI-AUTOMÁTICO INICIALIZADO POR SEMENTE E CORREÇÃO DO TRANSDUTOR)
# realizando o crescimento de região inicializado por semente
canva = np.zeros((M, N), np.uint8)
canva = skimage.img_as_float(canva)
grow = bibGrow.regionGrow(flong, canva, Cmin, Lmin, Cmax, Lmax, M, N, 3)

# correção do catéter
for l in range(M):
    for c in range(N):
       distx = c - 200
       disty = l - 200
       dist = np.math.sqrt(distx**2 + disty**2)
       if dist <= 40:
           grow[l,c] = 1

# plotando a imagem
plt.figure()
plt.title('Crescimento de região')
plt.imshow(grow, cmap='gray')

#%% PÓS-PROCESSAMENTO (FECHAMENTO E DILATAÇÃO)
# operação de fechamento
so = skimage.morphology.disk(40)
closed = skimage.morphology.binary_closing(grow, so)

# operação de dilatação
sd = skimage.morphology.disk(8)
final = skimage.morphology.dilation(closed, sd)

# plotando a imagem
plt.figure()
plt.title('Objeto segmentado final')
plt.imshow(final, cmap='gray')

# mostrando resultados
resultadoFinal = bibSeg.fazerAva(final, GS)   
print('parâmetros de avaliação ([VP, FP, FN, OD, OR]):', resultadoFinal)