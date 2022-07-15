# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:08:40 2022

@author: gjy3r6 (Elton Landers)
"""
from PIL import Image
import os, sys



dir_path = r"C:\Users\gjy3r6\Documents\Datasets\sample"

for file in os.listdir(dir_path):
    f_img = dir_path + '/' + file
    try:
        img = Image.open(f_img)
        img = img.resize((656, 504))
        img.save(f_img)
    except IOError:
        pass
    