# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:55:09 2020

@author: elton
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np

#LINE ON AN IMAGE
image_path=('beach.jpg')
image=cv2.imread(image_path)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

#set start and end co-ordinates
start=(0, 0)
end=(image.shape[1], image.shape[0])
color=(0,0,255)
thickness=4
cv2.line(image, start, end, color, thickness)

plt.imshow(image)

#RECTANGLE ON AN IMAGE
image_path=('beach.jpg')
image=cv2.imread(image_path)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

start=(75, 275) #x,y
end=(400, 375)
color=(0, 255, 0)
thickness=5
cv2.rectangle(image, start, end, color, thickness)

plt.imshow(image)

#DRAWING A RECTANGLE ON A NEW CANVAS AND SAVING THE IMAGE
#create a new canvas
canvas=np.zeros((200, 200, 3), dtype='uint8')
start=(10, 10)
end=(100, 100)
color=(255, 0, 0)
thickness=-1 #fill
cv2.rectangle(canvas, start, end, color, thickness)
cv2.imwrite('rectangle.jpg', canvas)
plt.imshow(canvas)

#CIRCLE ON AN IMAGE
canvas=np.zeros((200, 200, 3), dtype='uint8')
center=(100, 100)
radius=50
color=(255, 0, 0)
thickness=5
cv2.circle(canvas, center, radius, color, thickness)
plt.imshow(canvas)

center=(100, 100)
radius=75
color=(255, 0, 0)
thickness=5
cv2.circle(canvas, center, radius, color, thickness)
plt.imshow(canvas)
    
