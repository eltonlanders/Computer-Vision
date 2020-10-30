# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:52:59 2020

@author: elton
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt

image_path=('beach.jpg')
image=cv2.imread(image_path)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

#access pixel at (0, 0) location
(r, g, b)=image[0, 0]
print('Red, Green and Blue values at (0, 0):', format((r, g, b)))

#manipulate pixels and show modified image
image[0:100, 0:100]=(0, 255, 255) #pure blue, pure green
plt.imshow(image)