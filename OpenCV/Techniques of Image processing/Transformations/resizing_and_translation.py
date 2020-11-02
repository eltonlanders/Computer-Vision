# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:47:51 2020

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

#calc aspect ratio
(h,w)=image.shape[:2]
aspect=w/h

#decreasing height by half of the original image
height=int(0.5*h)
width=int(height*aspect)

#new image dim
dimension=(height, width)
resizedImage=cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
plt.imshow(resizedImage)

#resize using x and y factors
resizedwithfactors=cv2.resize(image, None, fx=1.2, fy=1.2, #resize factors
                              interpolation=cv2.INTER_LANCZOS4)
plt.imshow(resizedwithfactors)

#TRANSLATION
image_path=(r'C:/Users/elton/Documents/Computer Vision/Building computer vision applications using ANN/Core concepts of image  and video processing/beach.jpg')
image=cv2.imread(image_path)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

#defining translation matrix to move image to the right and down
translationMatrix=np.float32([[1, 0, 50], [0, 1, 20]])

#moving the image
movedImage=cv2.warpAffine(image, translationMatrix, (image.shape[1], image.shape[0]))

plt.imshow(movedImage)

#defining translation matrix to move image to the left and up
translationMatrix=np.float32([[1, 0, -50], [0, 1, -60]])

#moving the image
movedImage2=cv2.warpAffine(image, translationMatrix, (image.shape[1], image.shape[0]))

plt.imshow(movedImage2)

