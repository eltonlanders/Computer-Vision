# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:52:19 2020

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

#median filtering for noise reduction
blurredImage1=cv2.medianBlur(image1, 3)
plt.imshow(blurredImage1)

blurredImage2=cv2.medianBlur(image1, 5)
plt.imshow(blurredImage2)

