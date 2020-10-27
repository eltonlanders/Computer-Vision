# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:10:56 2020

@author: elton
"""

from __future__ import print_function
import argparse
import imutils
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
#cv.namedWindow('output', cv.WINDOW_NORMAL)

args={
    "image":"C:/Users/elton/Pictures/Photos/claudio-schwarz-purzlbaum-1ytdHHonH44-unsplash.jpg"
}
image=cv.imread(args['image'])
image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(image)

"""
AVERAGING
A convolutional kernel or simply a kernal is a sliding window applied on the image from top to bottom and left to right
The greater the size of the kernal the greater is the blurring
Uses simple averaging
"""
blurred=np.hstack([               #hstack stacks our output images together 
    cv.blur(image, (3, 3)),
    cv.blur(image, (15, 15)),   #value must be an odd number 
    cv.blur(image, (199, 199))])
plt.imshow(blurred)

"""
GAUSSIAN
Uses weighted mean
Less blurring comparatively, but more natural
"""
blurred=np.hstack([               #hstack stacks our output images together 
    cv.GaussianBlur(image, (3, 3), 0),   #last parameter is the std deviation, and 0 signifies to automatically compute them based on the kernel size
    cv.GaussianBlur(image, (15, 15), 0),   #value must be an odd number 
    cv.GaussianBlur(image, (199, 199), 0)])
plt.imshow(blurred)

"""
MEDIAN
Most effective to remove salt and pepper noise
Here instead of mean, median is used to replace
This reduces noise as the pixel we are representing must exist in our neighborhood
"""
blurred=np.hstack([               #hstack stacks our output images together 
    cv.medianBlur(image, 3),
    cv.medianBlur(image, 15),   #value must be an odd number 
    cv.medianBlur(image, 199)])
plt.imshow(blurred)               #Not getting a motion blur, but removing deta

"""
BILATERAL
To redeuce noise and still maintaining edges, use this technique
uses two gaussian distributions
slower than averaging
"""
blurred=np.hstack([               #hstack stacks our output images together 
    cv.bilateralFilter(image, 3, 21, 21),   #3 is the diameter of the neighborhood, 21 is the color(higher color means more color will be considered), 21 is the space,
    cv.bilateralFilter(image, 7, 31, 31),   #larger value of space means the pixels farther out from the central pixel will influence the blurring calculation 
    cv.bilateralFilter(image, 9, 41, 41)])
plt.imshow(blurred)               
