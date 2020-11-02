# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 07:15:08 2020

@author: elton
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1_path=(r'C:/Users/elton/Documents/Computer Vision/Building computer vision applications using ANN/Techniques of image processing/Image arithmetics and bitwise operations/image1.jpg')
image1=cv2.imread(image1_path)
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1=cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
image1=cv2.bilateralFilter(image1, 5, 50, 50)
plt.imshow(image1, cmap='gray')

#sobel gradient detection
sobelx = cv2.Sobel(image1,cv2.CV_64F,1,0,ksize=3)
sobelx=np.uint8(np.absolute(sobelx))
sobely=cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
sobely=np.uint8(np.absolute(sobely))

plt.imshow(sobelx)
plt.imshow(sobely)

#Schar gradient detection by passing ksize = -1 to Sobel function
scharx=cv2.Sobel(image1,cv2.CV_64F,1,0,ksize=-1)
scharx=np.uint8(np.absolute(scharx))
schary=cv2.Sobel(image1,cv2.CV_64F,0,1,ksize=-1)
schary=np.uint8(np.absolute(schary))

plt.imshow(scharx)
plt.imshow(schary)




