from mainUI import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QEvent, QObject
import os
import numpy as np
import requests
from PyQt5.QtCore import QBasicTimer,QThread, pyqtSignal
import json
import base64
import cv2
from PyQt5.QtWidgets import QApplication

UPLOAD_API_URL = 'http://166.111.82.233:8080/upload'
DETECT_API_URL = 'http://166.111.82.233:8080/detect'

class MainWindow(QMainWindow, Ui_MainWindow):
    signal_stop = pyqtSignal()
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        #qApp.installEventFilter(self)
        self.pathA = None       #上传图像的地址
        self.pathB = None
        self.rects = None       #返回结果[x1,y1,x2,y2,score,isEdge]
        self.rects_filtered = None  #过滤后的结果
        self.imgA = None        #配准后的img
        self.imgB = None
        self.imgA_tmp = None    #根据过滤后的rect，进行标记
        self.imgB_tmp = None    #根据过滤后的rect，进行标记
        self.threshold = None   #过滤结果阈值
        self.index = 0          #当前显示的Rect
        self.windowSize = 256 #局部显示窗口大小
        self.setupUi(self)
        self.bindEvents()
        self.show()

    def bindEvents(self):
        self.btn_selNegBoard.clicked.connect(self.selectImg)
        self.btn_selOkBoard.clicked.connect(self.selectImg)
        self.btn_upload.clicked.connect(self.uploadImg)
        self.btn_startDetect.clicked.connect(self.startDetect)
        self.horizontalSlider.valueChanged.connect(self.confChange)
        self.btn_filterRes.clicked.connect(self.confirmFiltRes)
        self.btn_moveLeft.clicked.connect(self.moveLeft)
        self.btn_moveRight.clicked.connect(self.moveRight)
        self.tabWidget.currentChanged['int'].connect(self.moveLeft)
        self.btn_enlarge.clicked.connect(self.enlarge)
        self.btn_shrink.clicked.connect(self.shrink)

        self.btn_upload.setEnabled(False)
        self.btn_startDetect.setEnabled(False)
        self.tabWidget.setTabEnabled(1,False)

    
    #放大缩小Patch
    def enlarge(self):
        self.windowSize = int(self.windowSize * 0.7)
        self.showRect()
    
    def shrink(self):
        self.windowSize = int(self.windowSize * 1.4)
        self.showRect()

    #向左向右翻页
    def moveLeft(self):
        self.index = max(0,self.index-1)
        self.lbe_indexNow.setText('%d / %d'%(self.index+1,len(self.rects_filtered)))
        self.showRect()

    def moveRight(self):
        self.index = min(len(self.rects_filtered)-1,self.index+1)
        self.lbe_indexNow.setText('%d / %d'%(self.index+1,len(self.rects_filtered)))
        self.showRect()
        
    #将筛选后的结果显示
    def confirmFiltRes(self):
        self.index = 0
        self.lbe_indexNow.setText('%d / %d'%(self.index+1,len(self.rects_filtered)))
        self.imgA_tmp = self.imgA.copy()
        self.imgB_tmp = self.imgB.copy()
        for x1,y1,x2,y2,score,isEdge in self.rects_filtered:
            if isEdge:
                color = (0,255,255)
            else:
                color = (0,0,255)
            self.imgA_tmp = draw_single_box(self.imgA_tmp,x1,y1,x2,y2,'%.3f'%(score),color)
            self.imgB_tmp = draw_single_box(self.imgB_tmp,x1,y1,x2,y2,'%.3f'%(score),color)
        cv2.imwrite('data/result/imgA_Rects.jpg',self.imgA_tmp)
        cv2.imwrite('data/result/imgB_Rects.jpg',self.imgB_tmp)
        self.showRect()

    
    #显示第index个rect
    def showRect(self):
        if len(self.rects_filtered) == 0:
            return None
        x1,y1,x2,y2,score,isEdge = self.rects_filtered[self.index,:]
        cx,cy = (x1+x2)//2,(y1+y2)//2
        H,W,_ = self.imgA.shape
        l = int(min(max(0,cx-self.windowSize//2),W-self.windowSize-1))
        t = int(min(max(0,cy-self.windowSize//2),H-self.windowSize-1))
        patchA = self.imgA_tmp[t:t+self.windowSize,l:l+self.windowSize,:]
        patchB = self.imgB_tmp[t:t+self.windowSize,l:l+self.windowSize,:]
        self.showArray(self.lbe_PatchA,patchA)
        self.showArray(self.lbe_PatchB,patchB)


    #根据置信度阈值更新筛选结果
    def filtRes(self):
        rects = self.rects
        selected = rects[rects[:,4] >= self.threshold]
        self.rects_filtered = selected
        num_edge = sum(selected[:,5] == 1)
        num_nonedge = sum(selected[:,5] == 0)
        self.lbe_defectCount.setText("缺陷总数(%d) = 边缘缺陷(%d) + 非边缘缺陷(%d)"%(len(selected),num_edge,num_nonedge))

    #更新置信度阈值显示
    def confChange(self,value):
        self.lbe_confThresh.setText('%.2f'%(value/100.0))
        self.threshold = value/100.0
        self.filtRes()

    #label控件显示path的图像
    def showImg(self,label,path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(label.size(),Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
    
    #显示Qimg图像
    def showQimg(self,label,qimg):
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(label.size(),Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    #显示ndarray图像
    def showArray(self,label,array):
        qimg = array2Qimage(array)
        self.showQimg(label,qimg)


    #选择图像
    def selectImg(self):
        path,_ = QFileDialog.getOpenFileName(self,"选取待检测PCB板图像",os.getcwd(),'Files (*.jpg)')
        sender = self.sender()
        if sender.text() == '选择待测试板':
            self.pathA = path
            self.showImg(self.lbe_imgA,path)
        elif sender.text() == '选择标准板': 
            self.pathB = path
            self.showImg(self.lbe_imgB,path)
        if self.pathA is not None and self.pathB is not None:
            self.btn_upload.setEnabled(True)
        self.statusBar().showMessage(sender.text()+' - 图片路径:' + path )
    

    #上传图像到服务器
    def uploadImg(self):
        if self.pathA is None or self.pathB is None:
            self.statusBar().showMessage('Error: 请选择待检测板和标准板！')
            return None

        self.statusBar().showMessage('正在上传图像...(由于网速限制20MB图像大概需要15s')
        self.uploader = Uploader(self.pathA,self.pathB)
        self.uploader.signal.connect(self.callback_upload)
        self.uploader.start()

        self.timer = Timer(15)
        self.timer.countChanged.connect(self.callback_updateBar)
        self.timer.start()
        self.signal_stop.connect(self.timer.stop)

    #发送检测信号
    def startDetect(self):
        self.tabWidget.setTabEnabled(1,False)
        self.detector = Detector()
        self.detector.signal.connect(self.callback_detect)
        self.detector.start()

        self.timer = Timer()
        self.timer.countChanged.connect(self.callback_updateBar)
        self.timer.start()
        self.signal_stop.connect(self.timer.stop)
        self.statusBar().showMessage('正在检测缺陷...')
    
 
    #========================== Call Back Functoins ============================================
    def callback_updateStatus(self):
        self.statusBar().showMessage(self.text)

    def callback_updateBar(self,value):
        self.progressBar.setValue(value)

    def callback_upload(self,flag):
        if flag:
            self.signal_stop.emit()
            time.sleep(0.8)
            self.progressBar.setValue(100)
            self.btn_startDetect.setEnabled(True)
            self.statusBar().showMessage('数据上传完毕!')
        else:
            self.signal_stop.emit()
            time.sleep(0.8)
            self.progressBar.setValue(0)
            self.statusBar().showMessage('Error: 数据上传错误!')
    
    def callback_detect(self,flag,rects,base64_imgA,base64_imgB):
        if flag:
            self.statusBar().showMessage('成功接收结果数据, 正在解码...')
            self.progressBar.setValue(92)
            QApplication.processEvents()
            self.signal_stop.emit()
            time.sleep(0.8)
            self.rects = np.array(rects)
            self.statusBar().showMessage('成功接收结果数据, 正在储存配准图像至 data/tmp...')
            self.progressBar.setValue(94)
            QApplication.processEvents()
            time.sleep(0.8)
            string2img(base64_imgA,'imgA.jpg')
            string2img(base64_imgB,'imgB.jpg')
            time.sleep(1)
            self.statusBar().showMessage('成功接收结果数据, 正在加载图像...')
            self.progressBar.setValue(96)
            QApplication.processEvents()
            time.sleep(0.8)
            self.imgA = cv2.imread('./data/tmp/imgA.jpg')
            self.imgB = cv2.imread('./data/tmp/imgB.jpg')
            QimgA = array2Qimage(self.imgA)
            QimgB = array2Qimage(self.imgB)
            self.statusBar().showMessage('成功接收结果数据, 正在显示图像...')
            self.progressBar.setValue(98)
            QApplication.processEvents()
            time.sleep(0.8)
            self.showQimg(self.lbe_imgA,QimgA)
            self.showQimg(self.lbe_imgB,QimgB)
            self.statusBar().showMessage('成功接收结果数据, 数据处理完毕!')
            self.progressBar.setValue(100)
            self.confChange(20)
            self.confirmFiltRes()
            self.tabWidget.setTabEnabled(1,True)
        else:
            if self.timer.isRunning():
                self.timer.terminate()
            self.progressBar.setValue(0)
            self.statusBar().showMessage('检测发生错误!')


import time

#状态栏线程
class Status(QThread):
    signal = pyqtSignal(str)
    def __init__(self,text,parent=None):
        super(Status,self).__init__(parent)
        self.text = text

    def run(self):
        self.signal.emit(self.text)
    
#进度条线程
class Timer(QThread):
    countChanged = pyqtSignal(int)
    def __init__(self,TimeLimit=5.0,parent=None):
        super(Timer,self).__init__(parent)
        self.det = 1
        self.TimeLimit = TimeLimit
        if TimeLimit == -1:
            self.det = 0
            self.TimeLimit = 1
        self.flag = True

    def run(self):
        count = 0
        while count < 90 and self.flag:
            count += self.det
            self.countChanged.emit(count)
            time.sleep(self.TimeLimit/100.0)
    
    def stop(self):
        self.flag = False
        
#上传线程
class Uploader(QThread):
    signal = pyqtSignal(bool)
    def __init__(self,pathA,pathB,parent=None):
        super(Uploader,self).__init__(parent)
        self.pathA = pathA
        self.pathB = pathB
    
    def run(self):
        imgA = open(self.pathA, 'rb').read()
        imgB = open(self.pathB, 'rb').read()
        payload = {'imgA': imgA,'imgB': imgB}
        r = requests.post(UPLOAD_API_URL, files=payload).json()
        if r['success']:
            self.signal.emit(True)
        else:
            self.signal.emit(False)

#检测线程
class Detector(QThread):
    signal = pyqtSignal(bool,list,str,str)   #[x1,y1,x2,y2,score,isEdge]*N,imgA,imgB
    def __init__(self,parent=None):
        super(Detector,self).__init__(parent)
    
    def run(self):
        payload = {'param': 'None'}
        r = requests.post(DETECT_API_URL,data=payload).json()
        base64_imgA = r['imgA']
        base64_imgB = r['imgB']
        rects = json.loads(r['result'])
        if r['success']:
            print('Success Detector!')
            self.signal.emit(True,rects,base64_imgA,base64_imgB)
        else:
            print('Request failed!')
            self.signal.emit(False,[],None,None)

#base64 --> img
def string2img(base64_img,name):
    image_data = base64.b64decode(base64_img)
    with open('./data/tmp/%s'%(name), 'wb') as jpg_file:
        jpg_file.write(image_data)


def array2Qimage(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res = QtGui.QImage(img.data, img.shape[1], img.shape[0],img.shape[1]*3,QtGui.QImage.Format_RGB888)
    return res

font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 0.6
#在ndarray上画rect
def draw_single_box(img,x1,y1,x2,y2,text=None,color=(0,0,200),color_text=(0,0,0),thickness=1):
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color,thickness)
    if text is not None:
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=2)[0]
        box_coords = ((x1, y1), (x1 + text_width + 4 , y1 - text_height -4))
        cv2.rectangle(img, box_coords[0], box_coords[1], [0,0,200], cv2.FILLED)
        cv2.putText(img,text,(int(x1),int(y1)-3),font,fontScale,color_text,thickness=2)
    return img

        

