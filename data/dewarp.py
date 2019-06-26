import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import cv2
import numpy as np
from tqdm import *

def show(res,name,K=5.0):
	res = (res-res.min())/(res.max()-res.min())
	res = torch.clamp(res,0.0,1.0/K)*K
	res = res[0,:,:,:].cpu().permute(1,2,0).numpy()*255
	cv2.imshow(name,res.astype(np.uint8))
		

class Dewarp(nn.Module):
	def __init__(self,maxDist = 7):
		super(Dewarp,self).__init__()
		self.maxDist = maxDist
		self.ks = 11
	
	def forward(self,A,B):
		D = self.maxDist
		A_pad = F.pad(A,[D,D,D,D])
		cx,cy = D,D
		_,_,H,W = B.size()	#A.size()==B.size()
		res = None
		weight_ = None
		for dx in tqdm(range(-D+1,D)):
			for dy in range(-D+1,D):
				px,py = cx+dx,cy+dy
				tmpRes = torch.norm(A_pad[:,:,py:py+H,px:px+W]-B,p=2,dim=1).unsqueeze(1)
				weight = F.avg_pool2d(tmpRes, self.ks, stride=1, padding=self.ks//2)
				#tmpRes = weight * tmpRes
				if weight_ is None or res is None:
					weight_ = weight
					res = tmpRes
				else:
					mask = weight_ >= weight 
					mask_ = weight[0,:,:,:].cpu().permute(1,2,0).numpy()*255
					#cv2.imshow('mask',mask_.astype(np.uint8))
					
					res[mask] = tmpRes[mask]
					weight_[mask] = weight[mask]
				res = res.detach()
				res_ = (res-res.min())/(res.max()-res.min())
				tmpRes= (tmpRes-tmpRes.min())/(tmpRes.max()-tmpRes.min())
				res_ = res_[0,:,:,:].cpu().permute(1,2,0).numpy()*255
				tmpRes = tmpRes[0,:,:,:].cpu().permute(1,2,0).numpy()*255
				cv2.imshow('Nowres',res_.astype(np.uint8))
				cv2.imshow('Tmpres',tmpRes.astype(np.uint8))
				cv2.waitKey(-1)
				weight_ = weight_.detach()
		res = F.max_pool2d(-res, 3, stride=1, padding=1)
		res = -res
		#return torch.mean(torch.abs(A-B),dim=1).unsqueeze(1)
		#return torch.norm(A-B,p=2,dim=1).unsqueeze(1)
		return res

if __name__ == '__main__':
	A = cv2.imread('A2.jpg').astype(np.float32)
	B = cv2.imread('B2.jpg').astype(np.float32)
	#cv2.imshow('A',A.astype(np.uint8))
	#cv2.imshow('B',B.astype(np.uint8))
	A = torch.from_numpy(A).permute(2,0,1)/255.0
	B = torch.from_numpy(B).permute(2,0,1)/255.0
	A = A.unsqueeze(0).float()
	B = B.unsqueeze(0).float()

	model = Dewarp(7)
	res = model(B,A)
	show(res,'res')
	cv2.waitKey(-1)




