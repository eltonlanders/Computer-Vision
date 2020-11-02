# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:15:29 2020

@author: elton
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1_path=('image1.jpg')
image2_path=('image2.jpg')

image1=cv2.imread(image1_path)
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
plt.imshow(image1)

image2=cv2.imread(image2_path)
image2=cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
plt.imshow(image2)

#resizing both the images to make them of the same dimension. This is a must
#to add two images
resizedImage1=cv2.resize(image1, (400, 400), interpolation=cv2.INTER_AREA)
resizedImage2=cv2.resize(image2, (400, 400), interpolation=cv2.INTER_AREA)

#this is a simple addition of two images
resultant=cv2.add(resizedImage1, resizedImage2)

plt.imshow(resizedImage1)
plt.imshow(resizedImage2)
plt.imshow(resultant)

#this is a weighted addition of two images
weightedImage=cv2.addWeighted(resizedImage1, 0.7, resizedImage2, 0.3, 0)
plt.imshow(weightedImage)

imageEnhanced=255*resizedImage1
plt.imshow(imageEnhanced)

arrayimage=resizedImage1 + resizedImage2
plt.imshow(arrayimage)

