# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:31:47 2020

@author: elton
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1_path=(r'C:/Users/elton/Documents/Computer Vision/Building computer vision applications using ANN/Techniques of image processing/Image arithmetics and bitwise operations/image1.jpg')
image1=cv2.imread(image1_path)
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
plt.imshow(image1)

#split images into component colors
(r, g, b)=cv2.split(image1)

#show the blue image
plt.imshow(b)

#show the red image
plt.imshow(r)

#show the green image
plt.imshow(g)

merged=cv2.merge([r, g, b])
plt.imshow(merged)