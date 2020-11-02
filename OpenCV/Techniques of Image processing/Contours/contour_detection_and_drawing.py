# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 07:36:59 2020

@author: elton
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1=cv2.imread('sudoku.png')
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1=cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
#image1=cv2.bilateralFilter(image1, 5, 50, 50)
plt.imshow(image1, cmap='gray')

#binarize the image
(T,binarized) = cv2.threshold(image1, 0, 255,
                              cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.imshow(binarized)

#canny function for edge detection
canny=cv2.Canny(binarized, 0, 255)
plt.imshow(canny)

_, contours, _= cv2.findContours(canny, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
print('Number of contours determined are', format(len(contours)))

copiedImage=image1.copy()
cv2.drawContours(copiedImage, contours, -1, (0, 255, 0), 2)
plt.imshow(copiedImage)



