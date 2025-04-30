import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import open3d as o3d
import math 
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import cv2  
import time 
from dataset import PCDDATATRAIN , AngleData 
from model import AngleNetwork





train_pcd = "/media/parvez/Expansion/TiHAN_maini/testbed_maingate/pcd" 
train_camera = "/media/parvez/Expansion/TiHAN_maini/testbed_maingate/camera" 
angle_path = "/media/parvez/Expansion/temp_data/angle.json" 




train_ds = PCDDATATRAIN(pcd_path=train_pcd, camera_path=train_camera, angle_path=angle_path, train=False)
train_loader = DataLoader(dataset = train_ds, batch_size=1, shuffle=False) 


device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Network2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
              
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out 
    


class Network1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.fc1 = nn.Linear(896, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
              
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out 
    


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network1 = Network1()
        self.network2 = Network2()
        
        self.fc1 = nn.Linear(20, 12)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(12, 6)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(6, 1)
        
    def forward(self, x1, x2):
        
        out1 = self.network1(x1)
        out2 = self.network2(x2)
        
        out = torch.cat((out1, out2), dim=1)
        
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        
        output = self.fc3(out)
        
        
        return output 
    






train_dataset = AngleData(pcd_path=train_pcd, camera_path=train_camera, angle_path=angle_path, train=False)
train_loader_our = DataLoader(dataset = train_dataset, batch_size=1, shuffle=False) 

loaded_checkpoint = torch.load("angle.pth") 
model_parameters = loaded_checkpoint["model_state"]
torch.save(model_parameters, "model_state_angle.pth") 


model_our = AngleNetwork()
model_our.load_state_dict(torch.load("model_state_angle.pth"))
model_our.to(device)
model_our.eval() 





loaded_checkpoint = torch.load("comparison.pth") 
model_parameters = loaded_checkpoint["model_state"]
torch.save(model_parameters, "model_state_comparison.pth") 


model = Model()
model.load_state_dict(torch.load("model_state_comparison.pth"))
model.to(device) 
model.eval() 

optimizer = torch.optim.SGD(model.parameters(), lr=0.001) 

mseloss = nn.L1Loss() 

training_loss = [] 

def train(model, train_loader, epoch):
    start_time = time.time()
    for i in range(epoch):
        running_loss = 0
        
        for n, data in enumerate(train_loader):
            
            
            pcd1 = data['pcd1'].to(device).float()
            pcd2 = data['pcd2'].to(device).float()
            gt = data['angle'].to(device).float()
            
            pcd1 = pcd1.unsqueeze(dim=1)
            pcd2 = pcd2.unsqueeze(dim=1)
            gt = gt.unsqueeze(dim=1)
            
            predict = model(pcd1, pcd2)
            
            loss = mseloss(predict, gt)
            
            
            optimizer.zero_grad()
            loss.backward()
            model.float()
            optimizer.step()
            
            
            running_loss += loss.item() 

            #print("hello, ", n)

        #print("time = ", time.time() - start_time)
            
        print("running_loss = {} epoch={}".format(running_loss, i+1))
        training_loss.append(running_loss)

        checkpoint={
            "epoch_number": i+1,
            "model_state": model.state_dict()
        }
        torch.save(checkpoint, "comparison.pth") 
        



predicted_angle = []  
gt_angle = []  
pred_angle = [] 


def validation(model, valid_loader):
    
    for n, data in enumerate(valid_loader):
        
            pcd1 = data['pcd1'].to(device).float()
            pcd2 = data['pcd2'].to(device).float()
            gt = data['angle'].to(device).float()
            
            pcd1 = pcd1.unsqueeze(dim=1)
            pcd2 = pcd2.unsqueeze(dim=1)
            gt = gt.unsqueeze(dim=1)
            
            predict = model(pcd1, pcd2).squeeze()
            
            predicted_angle.append(predict.item())
            



def eval(model):
    for n, data in enumerate(train_loader_our): 
          voxels = data["voxels"].float().to(device)
          image = data["image"].float().to(device)
          angle = data["angle"].float().to(device) 
          out = model(voxels, image) 
          gt_angle.append(angle.item())
          pred_angle.append(out.item())   
       

eval(model_our) 

#train(model, train_loader, 40) 


#plt.plot(np.arange(len(training_loss))+1, training_loss)
#plt.show()

validation(model, train_loader)

error_sum = 0 
for i in range(len(predicted_angle)):
    error_sum = error_sum + (predicted_angle[i] - gt_angle[i])**2

print("RMSE=", math.sqrt(error_sum / len(predicted_angle))) 




#plt.plot((np.arange(len(gt_angle))+1), gt_angle, label="ground truth steering angle")
#plt.plot((np.arange(len(pred_angle))+1), pred_angle, label="predicted steering angle (Our method)") 
#plt.plot((np.arange(len(pred_angle))+1), predicted_angle, label="predicted steering angle (LiDAR based method)")
#plt.xlabel("Sensor Frames") 
#plt.ylabel("Steering Angle Values")
#plt.legend() 
#plt.show() 










