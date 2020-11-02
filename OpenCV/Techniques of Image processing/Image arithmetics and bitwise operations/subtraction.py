# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:26:13 2020

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

#resize the two images to make them of the same dimensions. 
#This is a must to subtract two images
resizedImage1 = cv2.resize(image1,(int(500*image1.shape[1]/image1. shape[0]),
                                   500),interpolation=cv2.INTER_AREA)
resizedImage2 = cv2.resize(image2,(int(500*image2.shape[1]/image2. shape[0]),
                                   500),interpolation=cv2.INTER_AREA)

plt.imshow(resizedImage1)
plt.imshow(resizedImage2)

#subtract image 1 from 2
sub=cv2.subtract(resizedImage2, resizedImage1)
plt.imshow(sub)

#subtract image 2 from 1
subtractedImage=cv2.subtract(resizedImage1, resizedImage2)
plt.imshow(subtractedImage)

#numpy subtraction image 2 from 1
subimage=resizedImage2 - resizedImage1
plt.imshow(subimage)

#a constant subtraction
subtractedImage3=resizedImage1 - 50
plt.imshow(subtractedImage3)