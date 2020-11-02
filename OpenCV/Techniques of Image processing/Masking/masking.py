# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:59:23 2020

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

#create a rectangular mask
maskImage = cv2.rectangle(np.zeros(image1.shape[:2], dtype="uint8"),
                           (50, 50), (int(image1.shape[1])-50,
                                      int(image1.shape[0] / 2)-50),
                           (255, 255, 255), -1)
plt.imshow(maskImage)

#Using bitwise_and operation perform masking. Notice the mask=maskImage argument
masked = cv2.bitwise_and(image1, image1, mask=maskImage)
plt.imshow(masked)



