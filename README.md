# PCB_Defect_Detection_GUI
## Introduction
- main.py
- mainUI.py
  
  Program UI interface configuration file.
- mainWindow.py
  
  Interactive logic code.

## Installation[python3.6]
### Requirements
- Python 3.6+
- PyQt5
- requests
- base64
- json
- opencv-python
- numpy

### Easy Install
```shell
pip install -r requirements.txt
```

## Get Started
```shell
python main.py
```

## Interface
- Step 1.Select test board and standard board.
![home](demo/home.png)
- Step 2.Upload image to server.
![upload](demo/upload.png)
- Step 3.Send detection signal.
![detect](demo/detect.png)
- Step 4.Select the display page.Set the threshold and observe the results.
![show](demo/show.png)
- Step 5.
  - Images after registration can be found in 'data/tmp/'.
  - Images with annotated results are stored 'data/result/'
