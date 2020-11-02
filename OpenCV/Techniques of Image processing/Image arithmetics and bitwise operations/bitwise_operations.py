# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:39:55 2020

@author: elton
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np

#create a circle
circle=cv2.circle(np.zeros((200, 200, 3), dtype='uint8'), (100, 100), 90,
                                (255, 255, 255), -1)
plt.imshow(circle)

#create a square
square = cv2.rectangle(np.zeros((200,200,3), dtype= "uint8"), (30,30),
                        (170,170),(255,255,255), -1)
plt.imshow(square)

#bitwise AND
bitwiseAnd = cv2.bitwise_and(square, circle)
plt.imshow(bitwiseAnd)

#bitwiseOR
bitwiseOr = cv2.bitwise_or(square, circle)
plt.imshow(bitwiseOr)

#bitwiseXOR
bitwiseXor = cv2.bitwise_xor(square, circle)
plt.imshow(bitwiseXor)

#bitwiseNOT
bitwiseNot = cv2.bitwise_not(square)
plt.imshow(bitwiseNot)