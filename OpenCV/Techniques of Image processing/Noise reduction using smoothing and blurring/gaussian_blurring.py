# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:45:46 2020

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

#Gaussian blurring with 3x3 kernel and 0 for standard deviation to calculate from the kernel
GaussianFiltered=cv2.GaussianBlur(image1, (5, 5), 0)
plt.imshow(GaussianFiltered)