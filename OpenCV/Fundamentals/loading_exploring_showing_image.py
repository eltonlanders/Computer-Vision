# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:39:53 2020

@author: elton
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt

image_path=('beach.jpg')
image=cv2.imread(image_path)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

print('Dimensions of the image:', image.ndim)
print('Image height:', format(image.shape[0]))
print('Image width:', format(image.shape[1]))
print('Image channels:', format(image.shape[2]))
print('Size of the image array:', image.size)

plt.imshow(image)