# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 11:46:07 2022

@author: thais
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import skimage.exposure

# leitura da imagem
Mamo = cv2.imread('img18.pgm', 0)
Stent = cv2.imread('img19.pgm', 0)

# normalização das imagens
MamoN = skimage.img_as_float(Mamo)
StentN = skimage.img_as_float(Stent)

# exibição da imagem
plt.figure()
plt.title('img1')
plt.imshow(MamoN, cmap='gray')
plt.figure()
plt.title('img2')
plt.imshow(StentN, cmap='gray')

# encontrando o tamanho MxN da imagem e imprimindo
(M,N) = np.shape(Mamo)
(Ms, Ns) = np.shape(Stent)

# media das regiões
roim = cv2.selectROI(MamoN)
Cmin = roim[0]
Lmin = roim[1]
Cmax = roim[0] + roim[2]
Lmax = roim[1] + roim[3]  
mm = np.mean(MamoN[Lmin:Lmax, Cmin:Cmax])
dm = np.std(MamoN[Lmin:Lmax, Cmin:Cmax])

rois = cv2.selectROI(StentN)
Cmin = rois[0]
Lmin = rois[1]
Cmax = rois[0] + rois[2]
Lmax = rois[1] + rois[3] 
ms = np.mean(StentN[Lmin:Lmax, Cmin:Cmax])
ds = np.std(StentN[Lmin:Lmax, Cmin:Cmax])

# utilizando a imagem normalizada
histograma0 = skimage.exposure.histogram(MamoN)
x0 = histograma0[1]
y0 = histograma0[0]
plt.figure()
plt.stem(x0, y0, use_line_collection=True)
plt.title('Histograma Imagem0 Original')
plt.ylim([0, 5000])
plt.show()
histograma1 = skimage.exposure.histogram(StentN)
x1 = histograma1[1]
y1 = histograma1[0]
plt.figure()
plt.stem(x1, y1, use_line_collection=True)
plt.title('Histograma Imagem19 Original')
plt.ylim([0, 5000])
plt.show()

# alongando o contraste
long0 = skimage.exposure.rescale_intensity(MamoN, in_range=(0.02,0.73))
long19 = skimage.exposure.rescale_intensity(StentN, in_range=(0.02,0.73))

# plotando a imagem
plt.figure()
plt.title('img0 long')
plt.imshow(long0, cmap='gray')
plt.figure()
plt.title('img19 long')
plt.imshow(long19, cmap='gray')
 
#plotando o histograma
histogramaAlongada0 = skimage.exposure.histogram(long0)
x01 = histogramaAlongada0[1]
y01 = histogramaAlongada0[0]
plt.figure()
plt.stem(x01, y01, use_line_collection=True)
plt.title('Histograma img0 long')
plt.ylim([0, 5000])
plt.show()
histogramaAlongada19 = skimage.exposure.histogram(long19)
x11 = histogramaAlongada0[1]
y11 = histogramaAlongada0[0]
plt.figure()
plt.stem(x11, y11, use_line_collection=True)
plt.title('Histograma img19 long')
plt.ylim([0, 5000])
plt.show()