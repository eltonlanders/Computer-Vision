This is my repo for work done with regards to YOLOv5 which is SOTA at the time in Object Detection tasks. <br>
This is the official GitHub implementation for YOLOv5 - https://github.com/ultralytics/yolov5 <br>

Starting with the data, the main folder should contain two subfolders. These are "images" and "labels". <br>
Within both these folders, two more subfolders "train" and "val" should be present. <br>
The subfolder "images" should contain images split into train and val sets. <br> 
The subfolder "labels" should contain labels split into train and val sets. There should be as many labels in train subfolder as there are images in the images train subfolder. It's the same for val subfolder. <br>
Each image will have its correct label within the correct folder. <br>
![snip-211](https://user-images.githubusercontent.com/57378191/185542960-a126e6b8-b845-4146-8efc-b9aadbd4a075.PNG)


The "label" should be a text file with the .txt extension. <br>
Every row is an object. <br>
The format of each row is - class x_center y_center width height <br>
To normalize xywh (mandatory), divide x_center and width by image width and divide y_center and height by image height. <br>
An example of a row (object) in the text file - 0 0.41768292682926833 0.3075396825396825 0.08231707317073171 0.05555555555555555 <br>

Tool used for labelling - https://pypi.org/project/labelme/ <br>
The tool outputs a JSON file with the correct annotation information. Convert this JSON to the required text format using the code "json_to_text.py". <br>

Edit the .yaml file to contain the train and val image paths. Enter the number of classes and the class names. <br>
Check out the reference .yaml file. <br>

Now use the "start_training.py" as a reference to start the training. <br>



