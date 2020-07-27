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


import argparse
import cv2
import numpy as np
import os
import time
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torchvision.transforms
from tqdm import tqdm


from loader.dataset import DataSet
from model import Net
from utils.plot_me import plot
from utils.utils_model import initialize_weights

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.005, help='learning rate')
    parser.add_argument('--epochs', default=100, help='epochs')
    parser.add_argument('--batch-size', default=16, help='batch size')
    parser.add_argument('--data-path', default='./data_gen/',help='dataset path')
    parser.add_argument('--pre-trained', default=False,type=str2bool, help='use pre trained model')
    parser.add_argument('--pre-trained-path',help='pre trained model path')
    parser.add_argument('--parallel',default=False,type=str2bool,help='Set to True to train on parallel GPUs')
    parser.add_argument('--beta1',default=0.9,help='Beta Values for Adam Optimizer')
    parser.add_argument('--beta2',default=0.999,help='Beta Values for Adam Optimizer')
    parser.add_argument('--log',default=True,type=str2bool,help='Set to False to stop logging')
    parser.add_argument('--save-path',default="./model_save",help='Save Model')
    parser.add_argument('--testing',default=True,type=str2bool,help='To test or not to test')
    return parser.parse_args()

pre=0
def scheduler(epoch):
    global pre
    if pre:
        return 1

    if epoch<=1:
        return 0.6 
    
    else:
        return 1 

def clear(): 
  
    if os.name == 'nt': 
        _ = os.system('cls') 

    else: 
        _ = os.system('clear')         


def train(model,batch_size, first,epochs, train_data,test_data, optimizer,save_path,log,testing):
    global pre
    
    if pre:
        testing=False

    if save_path[-1]!='/':
        save_path=save_path+'/'

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=64)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers=64)
    
    
    loss_grid= nn.MSELoss()
    loss_edge=nn.BCEWithLogitsLoss()
    
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

    if log:
        training_loss=[]
        validation_loss=[]
    
    if testing and not pre:
        first=-1
        print("\n \nSTARTING PRELIMINARY TESTS------------------\n Model starts training after test")
        time.sleep(5)
        
    loss_train=[]
    loss_val=[]
    for epoch in range(first+1,epochs):        
        print("Epoch ",epoch)
        model.train()
        if not testing:
            print("Learning Rate",optimizer.param_groups[0]['lr'])


        lamda=0.9
        train_samples=len(train_loader.dataset)
        loss_sum_train=0
        batches=int(len(train_loader.dataset)/batch_size)
        train_samples=batches*batch_size

        with tqdm(total=batches) as pbar:

            for inputs, edges,grid in train_loader:
                
                inputs, edges,grid= inputs.cuda(), edges.cuda(), grid.cuda()
                out1,out2=model(inputs.float())
                
                grid=grid.squeeze()
                optimizer.zero_grad()
                loss_grid=nn.MSELoss()(out1,grid)

                loss_output=loss_grid+(lamda*(nn.BCELoss()(out2.float(),edges.float())))
                loss_output.backward()
                optimizer.step()
                

                if testing:
                    pbar.update(batches)
                    break

                del grid,out1,out2,edges
                
                loss_sum_train += float(loss_grid)
                pbar.update(1)
        if not testing:
            loss_train.append(loss_sum_train/train_samples)
            print('Epoch {}, Training Loss{:.9f}'.format(str(epoch), (loss_sum_train/train_samples)))   
        
        
        with torch.no_grad():
            model.eval()
            test_samples=len(test_loader.dataset)
            batches=int(test_samples/batch_size)
            test_samples=batches*batch_size
            loss_sum_val=0

            with tqdm(total=batches) as pbar:
                for inputs, edges,grid in test_loader:
                    if testing:
                        pbar.update(batches)
                        break

                    inputs, edges,grid= inputs.cuda(), edges.cuda(), grid.cuda()
                    
                    outputs= model(inputs)
                    output=outputs[0].squeeze()
                    grid=grid.squeeze()
                    loss_grid=nn.MSELoss()(output,grid)
                    loss_sum_val += float(loss_grid)
                    
                    pbar.update(1)

            
            if not testing:    
                print('Epoch {}, Validation Loss{:.9f}'.format(epoch,(loss_sum_val/test_samples)))
                loss_val.append(loss_sum_val/test_samples)

        lr_scheduler.step()

        if log and not testing:
                with open(save_path + 'loss.txt', 'a') as f:
                    str1="Epoch  "+str(epoch)+"  Training Loss"+str((loss_sum_train/train_samples))+'\t'+"Val loss"+str((loss_sum_val/test_samples))+'\n'
                    f.write(str1)
                plot(loss_train,loss_val,save_path)
                torch.save(model.state_dict(), save_path + str(epoch) + '.pt')

        if testing:
            testing=False
            print("Testing Complete-----Beginning to train...")
            time.sleep(3)
            clear()

            continue


        
        
    
clear()
parser = get_args()
model_save_path =parser.save_path
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)
data_path = parser.data_path
assert data_path and os.path.isdir(data_path), 'Wrong Data path'

model = Net()

if parser.parallel:
    model=torch.nn.DataParallel(model).cuda()
else:
    model=model.cuda()
model.apply(initialize_weights)
print("Weights initialized by kaiming initialization")
optimizer = optim.Adam(model.parameters(), lr=float(parser.lr),betas=(float(parser.beta1),float(parser.beta2)))
first=0

if parser.pre_trained:
    pre=1
    assert os.path.exists(parser.pre_trained_path), 'Wrong path for pre-trained model'
    model_dict = model.state_dict()
    state_dict = torch.load(parser.pre_trained_path)
    
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict) 
    
    model.load_state_dict(state_dict)

    print(f'model {parser.pre_trained_path} loaded')
    path1=parser.pre_trained_path
    first=int(path1[path1.rindex("/")+1:path1.rindex(".")])
    print("Starting Training from {}".format(first))





transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])
    
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

dataset_train = DataSet(os.path.join(data_path, 'image'), os.path.join(data_path, 'label'), transform)
dataset_test = DataSet(os.path.join(data_path, 'image_test'), os.path.join(data_path, 'label'), transform)

if parser.testing:
	print("You have opted for tests")
train(model, int(parser.batch_size), first,int(parser.epochs), dataset_train,dataset_test, optimizer, parser.save_path,parser.log,parser.testing)
