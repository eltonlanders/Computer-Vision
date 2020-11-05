# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:59:17 2020

@author: elton
"""

import cv2
import skimage.feature as sk
import numpy as np
import matplotlib.pyplot as plt

image1_path=(r'C:/Users/elton/Documents/Computer Vision/Building computer vision applications using ANN/Techniques of image processing/Image arithmetics and bitwise operations/image1.jpg')
image1=cv2.imread(image1_path)
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1=cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
plt.imshow(image1, cmap='gray')

#calculate GLCM of the grayscale image
glcm=sk.greycomatrix(image1, [2], [0, np.pi/2])
print(glcm)

#calculate contrast
contrast=sk.greycoprops(glcm) #default is contrast
print('Contrast:', contrast)

#calculate dissimilarity
dissimilarity=sk.greycoprops(glcm, prop='dissimilarity')
print('Dissimilarity:',dissimilarity)

#calculate homogeneity
homogeneity=sk.greycoprops(glcm, prop='homogeneity')
print('homogeneity:',homogeneity)

#calculate ASM
ASM=sk.greycoprops(glcm, prop='ASM')
print('ASM:',ASM)

#calculate energy
energy=sk.greycoprops(glcm, prop='energy')
print('energy:',energy)

#calculate correlation
correlation=sk.greycoprops(glcm, prop='correlation')
print('correlation:',correlation)