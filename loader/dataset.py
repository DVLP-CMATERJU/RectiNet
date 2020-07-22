"""
Official Code Implementation of:
"A Gated and Bifurcated Stacked U-Net Module for Document Image Dewarping"
Authors:    Hmrishav Bandyopadhyay,
            Tanmoy Dasgupta,
            Nibaran Das,
            Mita Nasipuri

Code: Hmrishav Bandyopadhyay

Code references:
>>>https://github.com/wuleiaty/DocUNet
"""


import cv2
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms



class DataSet(Dataset):
	
	def __init__(self, images_folder, labels_folder, transform):
		
		assert os.path.exists(images_folder), 'Images folder does not exist'
		assert os.path.exists(labels_folder), 'Labels folder does not exist'
		self.images_folder=os.path.abspath(images_folder)
		self.labels_folder=os.path.abspath(labels_folder)
		self.images = os.listdir(self.images_folder)
		self.labels = os.listdir(self.labels_folder)

		self.transform = transform
		

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		
		img_filename = self.images[index]
		filename = img_filename[:img_filename.index('.')]
		label_filename = filename + '.npz'
		
		img = cv2.imread(self.images_folder + '/' + img_filename)
		label = np.load(self.labels_folder + '/' + label_filename)
		
		img_h=img.shape[0]
		img_w=img.shape[1]
		
		grid=torch.Tensor(label['grid'])
		grid = grid.view(img_h, img_w,2)
		

		img= cv2.resize((cv2.pyrDown(img)),(256,256))
		
		borders=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(borders,100,200)
		edges[edges==255]=1	
		
		if self.transform:
			img = self.transform(img)
			edges=self.transform(edges)
		
		return img,edges,grid


