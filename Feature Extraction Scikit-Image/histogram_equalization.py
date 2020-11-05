# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:47:21 2020

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
plt.imshow(image1, cmap='gray')

#calculate histogram
hist=cv2.calcHist([image1], [0], None, [256], [0, 255])

#plot histogram graph
plt.figure()
plt.title('Grayscale histogram')
plt.xlabel('bins')
plt.ylabel('number of pixels')
plt.plot(hist)
plt.show()

equalizedImage=cv2.equalizeHist(image1)
plt.imshow(equalizedImage, cmap='gray')

#calculate histogram of the equalized image
hist=cv2.calcHist([equalizedImage], [0], None, [256], [0, 255])

#plot histogram graph
plt.figure()
plt.title('Grayscale histogram')
plt.xlabel('bins')
plt.ylabel('number of pixels')
plt.plot(hist)
plt.show()

