# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 08:53:54 2020

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

colors=('red', 'green', 'blue')
#calculate histogram
for i, color in enumerate(colors):
    hist=cv2.calcHist([image1], [i], None, [32], [0, 256])
    #plot histogram
    plt.plot(hist, color=color)
    
plt.title('RGB color histogram')
plt.xlabel('bins')
plt.ylabel('number of pixels')
plt.show()

#exercise
maskImage = cv2.rectangle(np.zeros(image1.shape[:2], dtype="uint8"),
                           (50, 50), (int(image1.shape[1])-50,
                                      int(image1.shape[0] / 2)-50),
                           (255, 255, 255), -1)
plt.imshow(maskImage)

masked = cv2.bitwise_and(image1, image1, mask=maskImage)
plt.imshow(masked)

colors=('red', 'green', 'blue')
#calculate histogram
for i, color in enumerate(colors):
    hist=cv2.calcHist([masked], [i], None, [32], [0, 256])
    #plot histogram
    plt.plot(hist, color=color)
    
plt.title('RGB color histogram')
plt.xlabel('bins')
plt.ylabel('number of pixels')
plt.show()