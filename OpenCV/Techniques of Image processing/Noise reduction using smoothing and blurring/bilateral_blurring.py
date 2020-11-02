# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:59:33 2020

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

#bilateral filtering
filtered_img5=cv2.bilateralFilter(image1, 5, 150, 50)
plt.imshow(filtered_img5)

#bilateral with kernel 7
filtered_img7=cv2.bilateralFilter(image1, 7, 160, 60)
plt.imshow(filtered_img7)

