# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:52:42 2020

@author: elton
"""

from __future__ import print_function
import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
%matplotlib qt
cv.namedWindow('output', cv.WINDOW_NORMAL)

args={
    "image":"C:/Users/elton/Pictures/Wallpapers/PC/85323.png"
}

image=cv.imread(args['image'])
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {}".format(image.shape[2]))

plt.imshow(image)

cv.imwrite("newimage.jpg", image)

image[0,0]=(0,0,255)

image[0,0]

corner=image[0:100,0:100]

image[0:100, 0:100]=(0,255,0)
canvas=np.zeros((300, 300, 3), dtype="uint8")
plt.imshow(canvas)

green=(0, 255, 0)
cv.line(canvas, (0,0), (300, 300), green)
plt.imshow(canvas)

red=(255, 0, 0)
cv.line(canvas, (300, 0), (0, 300), red, 3)
plt.imshow(canvas)

cv.rectangle(canvas, (10, 10), (60, 60), green)
plt.imshow(canvas)

cv.rectangle(canvas, (50, 200), (200, 225), red, 5)
plt.imshow(canvas)

blue=(0, 0, 255)
cv.rectangle(canvas, (200, 50), (225, 125), blue, -1)
plt.imshow(canvas)

canvas=np.zeros((300, 300, 3), dtype="uint8")
(centerX, centerY)= (canvas.shape[1]//2, canvas.shape[0]//2)

centerX

centerY

white=(255, 255, 255)
for r in range(0, 175, 25):
    cv.circle(canvas, (centerX, centerY), r, white)
    
plt.imshow(canvas)

for i in range(0, 25):
    radius=np.random.randto_bytest(5, high=200)
    color=np.random.randint(0, high=256, size=(3,)).tolist()
    pt=np.random.randint(0, high=300, size=(2,))
    cv.circle(canvas, tuple(pt), radius, color, -1)
plt.imshow(canvas)