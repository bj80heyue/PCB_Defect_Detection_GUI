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
import matplotlib.pyplot as plt

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
        
        self.banEdge = True     #是否屏蔽边缘
        self.confThresh = [0.12,0.18,0.27] + [1.0]
        self.confLevel = 1

        self.showMode = 'stat'  # stat | global | compare
        self.setupUi(self)
        self.bindEvents()
        self.show()

    def bindEvents(self):
        self.btn_selNegBoard.clicked.connect(self.selectImg)
        self.btn_selOkBoard.clicked.connect(self.selectImg)
        self.btn_upload.clicked.connect(self.uploadImg)
        self.btn_startDetect.clicked.connect(self.startDetect)
        #self.horizontalSlider.valueChanged.connect(self.confChange)
        #self.btn_filterRes.clicked.connect(self.confirmFiltRes)
        self.btn_moveLeft.clicked.connect(self.moveLeft)
        self.btn_moveRight.clicked.connect(self.moveRight)
        self.tabWidget.currentChanged['int'].connect(self.moveLeft)
        self.btn_enlarge.clicked.connect(self.enlarge)
        self.btn_shrink.clicked.connect(self.shrink)
        self.btn_saveRes.clicked.connect(self.saveRes)

        self.btn_upload.setEnabled(False)
        #self.btn_startDetect.setEnabled(False)
        #self.tabWidget.setTabEnabled(1,False)

        self.btn_edge_yes.setEnabled(False)
        self.btn_edge_no.clicked.connect(self.banChange)
        self.btn_edge_yes.clicked.connect(self.banChange)

        self.btn_level_2.setEnabled(False)
        self.btn_level_1.clicked.connect(self.levelChange)
        self.btn_level_2.clicked.connect(self.levelChange)
        self.btn_level_3.clicked.connect(self.levelChange)

        self.btn_show_1.setEnabled(False)
        self.btn_show_1.clicked.connect(self.showChange)
        self.btn_show_2.clicked.connect(self.showChange)
        self.btn_show_3.clicked.connect(self.showChange)

    
    #条件显示
    def conditionShow(self):
        mode = self.showMode
        print(mode)
        self.label_8.setText('待检测板Patch')
        self.label_9.setText('标准版Patch')
        if mode == 'stat':
            self.showPie()
            self.label_8.setText('边缘区域缺陷统计')
            self.label_9.setText('非边缘区域缺陷统计')
        elif mode == 'compare':
            #self.confirmFiltRes()
            self.showRect()
        else:
            self.showGlobal()

    def showGlobal(self):
        if len(self.rects_filtered) == 0:
            return None
        #self.imgA_Full = self.imgA.copy()
        self.imgA_Full = self.imgA_tmp.copy()

        for x1,y1,x2,y2,score,isEdge in self.rects_filtered:
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            R = 200
            H,W,_ = self.imgA.shape
            l = int(min(max(0,cx-R//2),W-R-1))
            t = int(min(max(0,cy-R//2),H-R-1))
            color = (0,0,255)
            self.imgA_Full = draw_single_box(self.imgA_Full,l,t,l+R,t+R,None,color,thickness=20)

        x1,y1,x2,y2,score,isEdge = self.rects_filtered[self.index,:]
        if isEdge:
            color = (0,255,255)
        else:
            color = (0,0,255)
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        R = 200
        H,W,_ = self.imgA.shape
        l = int(min(max(0,cx-R//2),W-R-1))
        t = int(min(max(0,cy-R//2),H-R-1))
        color = (255,0,255)
        imgA_Full = draw_single_box(self.imgA_Full,l,t,l+R,t+R,None,color,thickness=20)

        l = int(min(max(0,cx-self.windowSize//2),W-self.windowSize-1))
        t = int(min(max(0,cy-self.windowSize//2),H-self.windowSize-1))
        patchA = self.imgA_tmp[t:t+self.windowSize,l:l+self.windowSize,:]
        self.showArray(self.lbe_PatchA,patchA)
        self.showArray(self.lbe_PatchB,imgA_Full)

        
    #显示饼图
    def showPie(self):
        #获取数据
        data_edge = []
        data_noedge = []
        rects = self.rects
        rects_edge = rects[rects[:,5]==1]
        rects_noedge= rects[rects[:,5]==0]

        for i in range(len(self.confThresh)-1):
            conf_down = self.confThresh[i]
            conf_up= self.confThresh[i+1]
            data_edge.append(((rects_edge[:,4]>conf_down) & (rects_edge[:,4]<=conf_up)).sum())
            data_noedge.append(((rects_noedge[:,4]>conf_down) & (rects_noedge[:,4]<=conf_up)).sum())
        #labels = ['Low','Mid','High']
        labels_edge = ['A : %d'%(data_edge[0]),'B : %d'%(data_edge[1]),'C : %d'%(data_edge[2])]
        labels_noedge = ['A : %d'%(data_noedge[0]),'B : %d'%(data_noedge[1]),'C : %d'%(data_noedge[2])]

        ex = [0,0,0]
        ex[self.confLevel] = 0.4
        genPie(labels_edge,data_edge,ex,['gray','yellow','red'],'pie_edge.jpg')
        genPie(labels_noedge,data_noedge,ex,['gray','yellow','red'],'pie_noedge.jpg')
        self.showImg(self.lbe_PatchA,'data/tmp/pie_edge.jpg')
        self.showImg(self.lbe_PatchB,'data/tmp/pie_noedge.jpg')


    #边缘情况更改
    def banChange(self): 
        sender = self.sender()
        if sender.text() == '是':
            self.banEdge = True
            self.btn_edge_yes.setEnabled(False)
            self.btn_edge_no.setEnabled(True)
        else:
            self.banEdge = False
            self.btn_edge_yes.setEnabled(True)
            self.btn_edge_no.setEnabled(False)
        self.filtRes()
        self.confirmFiltRes()
        self.conditionShow()
    
    #过滤等级更改
    def levelChange(self):
        sender = self.sender()
        self.btn_level_1.setEnabled(True)
        self.btn_level_2.setEnabled(True)
        self.btn_level_3.setEnabled(True)
        if sender.text() == '低风险':
            self.confLevel = 0
            self.btn_level_1.setEnabled(False)
        elif sender.text() == '中等风险':
            self.confLevel = 1
            self.btn_level_2.setEnabled(False)
        elif sender.text() == '高风险':
            self.confLevel = 2
            self.btn_level_3.setEnabled(False)
        self.filtRes()
        self.confirmFiltRes()
        self.conditionShow()
        
    #显示模式更改
    def showChange(self):
        sender = self.sender()
        self.btn_show_1.setEnabled(True)
        self.btn_show_2.setEnabled(True)
        self.btn_show_3.setEnabled(True)

        self.btn_moveLeft.setEnabled(True)
        self.btn_moveRight.setEnabled(True)
        self.btn_shrink.setEnabled(True)
        self.btn_enlarge.setEnabled(True)
        if sender.text() == '统计视图':
            self.showMode = 'stat'
            self.btn_show_1.setEnabled(False)
            self.btn_moveLeft.setEnabled(False)
            self.btn_moveRight.setEnabled(False)
            self.btn_shrink.setEnabled(False)
            self.btn_enlarge.setEnabled(False)
        elif sender.text() == '全局视图':
            self.showMode = 'global'
            self.btn_show_2.setEnabled(False)
            self.btn_shrink.setEnabled(False)
            self.btn_enlarge.setEnabled(False)
            #self.confirmFiltRes()
        elif sender.text() == '对比视图':
            self.showMode = 'compare'
            self.btn_show_3.setEnabled(False)
            #self.confirmFiltRes()
        self.conditionShow()


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
        self.conditionShow()
        #self.showRect()

    def moveRight(self):
        self.index = min(len(self.rects_filtered)-1,self.index+1)
        self.lbe_indexNow.setText('%d / %d'%(self.index+1,len(self.rects_filtered)))
        self.conditionShow()
        #self.showRect()
        
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
            self.imgA_tmp = draw_single_box(self.imgA_tmp,x1,y1,x2,y2,None,color)#'%.3f'%(score),color)
            self.imgB_tmp = draw_single_box(self.imgB_tmp,x1,y1,x2,y2,None,color)#'%.3f'%(score),color)
        #self.showRect()

    def saveRes(self):
        cv2.imwrite('data/result/imgA_Rects.jpg',self.imgA_tmp)
        cv2.imwrite('data/result/imgB_Rects.jpg',self.imgB_tmp)
        np.save('data/result/rects.npy',self.rects_filtered)
        self.statusBar().showMessage('结果图像保存完毕! 存放在data/result/imgX_Rects.jpg')
    
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
        #是否保留边缘
        if self.banEdge:
            selected = rects[rects[:,5]==0]
        else:
            selected = rects
        #获取区间
        conf_down = self.confThresh[self.confLevel]
        conf_up= self.confThresh[self.confLevel+1]
        selected = selected[selected[:,4] > conf_down]
        selected = selected[selected[:,4] <= conf_up]
        selected = selected[selected[:,4].argsort()[::-1]]
        self.rects_filtered = selected
        #num_edge = sum(selected[:,5] == 1)
        #num_nonedge = sum(selected[:,5] == 0)
        #self.lbe_defectCount.setText("缺陷总数(%d) = 边缘缺陷(%d) + 非边缘缺陷(%d)"%(len(selected),num_edge,num_nonedge))

    #更新置信度阈值显示
    def confChange(self,value):
        self.threshold = value/100.0
        self.filtRes()

    #label控件显示path的图像
    def showImg(self,label,path):
        pixmap = QPixmap(path)
        print(label.size())
        pixmap = pixmap.scaled(label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        label.setPixmap(pixmap)
    
    #显示Qimg图像
    def showQimg(self,label,qimg):
        pixmap = QPixmap.fromImage(qimg)
        size_ = label.size()
        size_img = QtCore.QSize(size_.width(),size_.height()-2)
        pixmap = pixmap.scaled(size_img,Qt.KeepAspectRatio)
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

        self.statusBar().showMessage('正在上传图像...(由于网速限制20MB图像大概需要15-30s')
        self.uploader = Uploader(self.pathA,self.pathB)
        self.uploader.signal.connect(self.callback_upload)
        self.uploader.start()

        self.timer = Timer(25)
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
            self.conditionShow()
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

#画饼图
def genPie(labels,data,explode,colors,name):
    #explods 分离度
    plt.figure(figsize=(8,8))
    plt.axes(aspect='equal')
    plt.xlim(0,8)
    plt.ylim(0,8)
    #if sum(data) == 0:
        #data[0] = 1
    patches,text,_ = plt.pie(x = data, # 绘图数据
        #labels=labels, # 添加编程语言标签
        explode=explode, # 突出显示Python
        colors=colors, # 设置饼图的自定义填充色
        autopct='%.3f%%', # 设置百分比的格式，此处保留3位小数
        pctdistance=0.8,  # 设置百分比标签与圆心的距离
        labeldistance = 1.15, # 设置标签与圆心的距离
        startangle = 180, # 设置饼图的初始角度
        center = (4, 4), # 设置饼图的圆心（相当于X轴和Y轴的范围）
        radius = 3.8, # 设置饼图的半径（相当于X轴和Y轴的范围）
        counterclock = False, # 是否逆时针，这里设置为顺时针方向
        #wedgeprops = {'linewidth': 1, 'edgecolor':'green'},# 设置饼图内外边界的属性值
        textprops = {'fontsize':17, 'color':'black'}, # 设置文本标签的属性值
        frame = 1) # 是否显示饼图的圆圈，此处设为显示
    # 不显示X轴和Y轴的刻度值
    plt.xticks(())
    plt.yticks(())
    plt.legend(patches, labels, loc="upper right",bbox_to_anchor=(1.1, 1.1),prop={'size': 15})
    plt.axis('off')
    plt.savefig('data/tmp/'+name,dpi=150)
        

import sys
if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = MainWindow()
	sys.exit(app.exec_())


