# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 14:51:19 2020

@author: elton
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path=(r'C:/Users/elton/Documents/Computer Vision/Building computer vision applications using ANN/Core concepts of image  and video processing/beach.jpg')
image=cv2.imread(image_path)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

(h, w)=image.shape[:2]

#defining translation matrix
center=(h//2, w//2)
angle=45
scale=0.5 #1.0 to keep original size

rotation_matrix=cv2.getRotationMatrix2D(center, angle, scale)

#rotate the image
rotatedImage=cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

plt.imshow(rotatedImage)

#FLIPPING
image_path=(r'C:/Users/elton/Documents/Computer Vision/Building computer vision applications using ANN/Core concepts of image  and video processing/beach.jpg')
image=cv2.imread(image_path)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

#flipping horizontally
flippedhorizontally=cv2.flip(image, 1)
plt.imshow(flippedhorizontally)

#flipping vertically
flipedvertically=cv2.flip(image, 0)
plt.imshow(flipedvertically)

#flipping horiontally and then vertically
flippedHV=cv2.flip(image, -1)
plt.imshow(flippedHV)