# Drone Following Me Assignment

## Norsang Nyandak

### How to install using Docker
Make sure to run the Docker file
    Command to build : ' docker build -t <image_name> . '
    Command to run : ' docker run <image_name>'



### Training and Labeling Images:

First make sure you have Anaconda downloaded or using python venv:
- Create a virtual environment using "conda create -n yolov8_custom python=3.12"
- Activate by "conda activate yolov8_custom"
Now to Train:
- Download specific images that you want to train the model with.
- Download labelImg using "pip install labelImg" in terminal.
- Now annotate the images by running the command "labelImg" in terminal.
    - Issues with labelImg : "float"
    - Solution: In canvas.py change these lines.
        "p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))

        p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))
        p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))" 
    - Rerun
- "pip install ultralytics"
-  Make sure to run using GPU not CPU
    - "python"
    - "import torch"
    - "torch.__version__" : Should be '...cu...'. CU = GPU and CPU = CPU
    - "torch.cudo.is_available()" : Should be True
-  Command "yolo task=detect mode=train epochs=100 data=data_custom.yaml model=yolov8m.pt imgsz=640" for yolov8m from their github.
        - Make sure your directory in data_custom.yaml has the right path for both .\train and .\val
-  Folder called runs is created that stores the model.
-  To retrain: Delete labeles.cache in both train and val folder.

Now to Predict:
- Inside runs>detect folder you can find the train folder that it suggested, copy and paste the best.pt inside the weight folder and copy in root
- Command : "yolo task=detect mode=predict model=Norsang_Model_V1.pt show=True conf=0.5 source=image.jpeg"
- For Videos: "yolo task=detect mode=predict model=Norsang_Model_V1.pt show=True conf=0.5 source="./Videos/Cyclist and vehicle Tracking - 1.mp4" "