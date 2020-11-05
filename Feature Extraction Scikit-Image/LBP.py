# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:58:17 2020

@author: elton
"""

import cv2
import skimage.feature as sk
import numpy as np
import matplotlib.pyplot as plt

image1_path=(r'C:/Users/elton/Documents/Computer Vision/Building computer vision applications using ANN/Techniques of image processing/Image arithmetics and bitwise operations/image1.jpg')
image1=cv2.imread(image1_path)
image1 = cv2.resize(image1, (int(image1.shape[0]/5), int(image1.shape[1]/5)))
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1=cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
plt.imshow(image1, cmap='gray')

#calculate Histogram of original image and plot it
originalHist = cv2.calcHist(image1, [0], None, [256], [0,256])

plt.figure()
plt.title('Histogram of original image')
plt.plot(originalHist, color='r')

#Calculate LBP image and histogram over the LBP, then plot the histogram
radius=3
points=3*8

#LBP calculation
lbp=sk.local_binary_pattern(image1, points, radius, method='default')
lbpHist, _ = np.histogram(lbp, density=True, bins=256, range=(0, 256))

plt.figure()
plt.title('histogram of lbp image')    
plt.plot(lbpHist, color='g')
plt.show

#showing the original and LBP images
plt.imshow(image1, cmap='gray')
plt.imshow(lbp, cmap='gray')


