"""
Official Code Implementation of:
"A Gated and Bifurcated Stacked U-Net Module for Document Image Dewarping"
Authors:    Hmrishav Bandyopadhyay,
            Tanmoy Dasgupta,
            Nibaran Das,
            Mita Nasipuri

Code: Hmrishav Bandyopadhyay

Code references:
>>>https://github.com/cvlab-stonybrook/DewarpNet
"""


import argparse
import cv2
import glob
import matplotlib.image as mpimg
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms

from model_pred	 import Net

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def unwarp(img, bm,img_scan):
    w,h=img_scan.shape[0],img_scan.shape[1]
    bm = bm.detach().cpu().numpy()[0,:,:,:]
    
    bm0=cv2.blur(bm[:,:,0],(3,3))
    bm1=cv2.blur(bm[:,:,1],(3,3))

    bm0=cv2.resize(bm0,(h,w))
    bm1=cv2.resize(bm1,(h,w))
    
    bm=np.stack([bm0,bm1],axis=-1)
    bm=np.expand_dims(bm,0)
    bm=torch.from_numpy(bm).double()


    
    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).double()

    res = F.grid_sample(input=img, grid=bm,align_corners=True)
    res = res[0].numpy().transpose((1, 2, 0))
    
    return res



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='model path')
    parser.add_argument('--img-path', help='image path or path to folder containing images (set multi as true)')
    parser.add_argument('--save-path', help='save path')
    parser.add_argument('--check', default=False ,type=str2bool,help='True if Checking for MS-SSIM on Scanned images')
    parser.add_argument('--scan-path', default="",help='Scanned Image path -- only if verifying MS-SSIM')
    parser.add_argument('--multi', default=False,type=str2bool ,help='True if predicting Multiple Images in same folder')
    return parser.parse_args()


def predict(model_path,img_path,save_path,scan_path,check,filename):
    
    assert os.path.exists(model_path), 'Incorrect Model Path'
    assert os.path.exists(img_path), 'Incorrect Image Path'
    assert os.path.exists(save_path), 'Incorrect Save Path'

    if check:
        assert os.path.exists(scan_path), 'Incorrect Scanned Images Path'

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]
    )
    
    img = cv2.imread(img_path)
    if check:
        scan_path=scan_path+img_path[img_path.rindex('/'):img_path.rindex('_')]+".png"
        img_scan=cv2.imread(scan_path,0)
    else:
        img_scan=img
    
    input_img=cv2.resize(img,(256,256))
    model = Net().cuda()
    model=torch.nn.DataParallel(model).cuda()
    
    assert os.path.exists(model_path), 'Wrong path for pre-trained model'
    model_dict = model.state_dict()
    state_dict = torch.load(model_path)
    
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict) 
    
    model.load_state_dict(state_dict)

    print(f'model {model_path} loaded')
    
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict) 
    
    model.load_state_dict(state_dict)

    model.eval()
    
    with torch.no_grad():
        input_ = transforms(input_img).cuda()
        input_ = input_.unsqueeze(0)
        output = model(input_)
    
    grid=output[0]
    image_unwarped=unwarp(img,grid.cpu(),img_scan)*255
    print(cv2.imwrite(filename,image_unwarped))
    

if __name__=='__main__':


    parser=get_args()
    
    model_path = parser.model_path
    img_path = parser.img_path
    save_path=parser.save_path
    check=parser.check
    scan_path=parser.scan_path
    multi=parser.multi
    if multi:
        print("Multi enabled")
        for file in glob.glob(img_path+"/*"):
            filename=(save_path+"/"+file[file.rindex("/")+1:file.rindex(".")]+"dewarp.png")
            predict(model_path,file,save_path,scan_path,check,filename)
            print("Written ",file[file.rindex("/")+1:file.rindex(".")])

    else:
        print("Multi Disabled")
        
        file=img_path   
        try:
            filename=(save_path+"/"+file[file.rindex("/")+1:file.rindex(".")]+"dewarp.png")
        except:
            filename=img_path[:img_path.rindex('.')]+"dewarp.png"
        predict(model_path,img_path,save_path,scan_path,check,filename)
        try:
            print("Written ",img_path[img_path.rindex("/")+1:img_path.rindex(".")])
        except:
            print("Written",img_path[:img_path.rindex('.')])


    
