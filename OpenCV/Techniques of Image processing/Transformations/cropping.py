# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:06:07 2020

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

#CROPPING
croppedImage=image[0:300, 475:600] #first y then for x
plt.imshow(croppedImage)