# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:57:15 2022

@author: Elton Landers
"""



# !git clone "https://github.com/ultralytics/yolov5"
pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

# Train command format
# !python <train.py path> --img <width of image> --batch <batch size> --epochs <# of epochs> --data <yaml file path> -- cfg <yolo model version> --name <name of the model>
!python train.py --img 656 --batch 2 --epochs 100 --data driving.yaml --cfg models/yolov5s.yaml --name custom_run_07 
