# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 16:37:14 2021

@author: gjy3r6 (Elton Landers)
"""
import os
import shutil
import glob



# The parent folder with subdirectories
parent_path = r"C:\Users\gjy3r6\Documents\Data\dataset_v3_reviewed\test\eyeNotDetected_unsure"

# Renaming the images with the folder name as prefix
for root, dirs, files in os.walk(parent_path):
    if not files:
        continue
    prefix = os.path.basename(root)
    for f in files:
        os.rename(os.path.join(root, f), 
                  os.path.join(root, "{}_{}".format(prefix, f)))


"""
for (root, dirs, files) in os.walk(parent_path):
    print(root)
    print(dirs)
    print(files)
    print('------------------------------')
"""


# Extracting all images in the single main folder
# folder = input("pick a folder : ")
subfolders = [f.path for f in os.scandir(parent_path) if f.is_dir()]

for sub in subfolders:
    for f in os.listdir(sub):
        src = os.path.join(sub, f)
        dst = os.path.join(parent_path, f)
        shutil.move(src, dst)
    shutil.rmtree(sub)


