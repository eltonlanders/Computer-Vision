# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:32:49 2020

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

#define the kernel
kernel=(3, 3)
blurred1=cv2.blur(image1, kernel)
plt.imshow(blurred1)

blurred2=cv2.blur(image1, (5, 5))
plt.imshow(blurred2)

blurred3=cv2.blur(image1, (7, 7))
plt.imshow(blurred3)
