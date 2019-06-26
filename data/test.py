import matplotlib.pyplot as plt
import cv2
import numpy as np
from dewarp import Dewarp
import torch

model = Dewarp(7)

imgA = cv2.imread('tmp/imgA.jpg')
imgB = cv2.imread('tmp/imgB.jpg')
rects = np.load('result/rects.npy')

for x1,y1,x2,y2,score,_ in rects[:25]:
	#cx = (x1+x2)//2
	#cy = (y1+y2)//2
	l = int(x1) 
	t = int(y1) 
	H = int(y2-y1)
	W = int(x2-x1)
	patchA = imgA[t:t+H,l:l+W,:]
	patchB = imgB[t-10:t+10+H,l-10:l+10+W,:]
	#mask = cv2.matchTemplate(patchB, patchA,cv2.TM_SQDIFF_NORMED)
	mask = cv2.matchTemplate(patchB, patchA,cv2.TM_CCOEFF_NORMED)
	print(mask.max())

	#res = np.hstack([patchA,patchB])
	plt.figure()
	plt.subplot(3,1,1)
	plt.imshow(patchA)
	plt.subplot(3,1,2)
	plt.imshow(patchB)
	plt.subplot(3,1,3)
	plt.imshow(mask)
	plt.show()
	


