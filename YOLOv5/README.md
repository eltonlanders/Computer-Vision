This is my repo for work done with regards to YOLOv5 which is SOTA at the time in Object Detection tasks. <br>
This is the official GitHub implementation for YOLOv5 - https://github.com/ultralytics/yolov5 <br>

Starting with the data, the main folder should contain two subfolders. These are "images" and "labels". <br>
Within both these folders, two more subfolders "train" and "val" should be present. <br>
The subfolder "images" should contain images split into train and val sets. <br> 
The subfolder "labels" should contain labels split into train and val sets. There should be as many labels in train subfolder as there are images in the images train subfolder. It's the same for val subfolder. <br>
Each image will have its correct label within the correct folder. <br>
Data -- Images
         -- train
           -- 1.png
           -- 2.png
         -- val
           -- 3.png
     -- Labels
         -- train
           -- 1.txt
           -- 2.txt
         -- val
           -- 3.txt


The label format 
