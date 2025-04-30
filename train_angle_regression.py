from PIL import Image
import requests
from transformers import AutoProcessor, FlavaModel
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import os 
import numpy as np 
from numba import njit 
import open3d as o3d 
from torch.utils.data import Dataset, DataLoader 
import json
import cv2  
from dataset import AngleData  
from model import AngleNetwork  
import matplotlib.pyplot as plt  
import numpy as np 
import math 



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  



train_pcd = "/media/parvez/Expansion/TiHAN_maini/testbed_maingate/pcd" 
train_camera = "/media/parvez/Expansion/TiHAN_maini/testbed_maingate/camera" 
angle_path = "/media/parvez/Expansion/temp_data/angle.json"


train_dataset = AngleData(pcd_path=train_pcd, camera_path=train_camera, angle_path=angle_path, train=False)
train_loader = DataLoader(dataset = train_dataset, batch_size=1, shuffle=False) 

loaded_checkpoint = torch.load("angle.pth") 
model_parameters = loaded_checkpoint["model_state"]
torch.save(model_parameters, "model_state_angle.pth") 


model = AngleNetwork()
model.load_state_dict(torch.load("model_state_angle.pth"))
model.to(device)
model.eval() 






optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) 

training_loss = []  



def train(model, epochs):
    for i in  range(epochs): 
       running_loss = 0.0 
       for n, data in enumerate(train_loader):
              voxels = data["voxels"].float().to(device)
              image = data["image"].float().to(device)
              angle = data["angle"].float().to(device) 
              out = model(voxels, image)
              loss = (out - angle)**2 
              # backprop 
              optimizer.zero_grad()
              loss.backward() 
              optimizer.step()

              #print(cosine_text_voxel, cosine_text_image)
              running_loss = running_loss + loss.item()  


       training_loss.append(running_loss)
       print("running_loss = {}, epoch={}".format(running_loss, i+1))

       checkpoint={
            "epoch_number": i+1,
            "model_state": model.state_dict()
        }
       torch.save(checkpoint, "angle.pth") 



#train(model, 100) 

gt_angle = []
pred_angle = []  


def eval(model):
    for n, data in enumerate(train_loader):  
          if n == 100:
             voxels = data["voxels"].float().to(device)
             image = data["image"].float().to(device)
             angle = data["angle"].float().to(device) 
             out = model(voxels, image) 
             gt_angle.append(angle.item())
             pred_angle.append(out.item())  
          else:
               continue 
            



     

eval(model) 
#error_sum = 0 
#for i in range(len(gt_angle)): 
#    error_sum = error_sum + (gt_angle[i] - pred_angle[i])**2 

#print("RMSE = ", math.sqrt(error_sum / len(gt_angle))) 



#plt.plot(np.arange(len(training_loss))+1, training_loss)
#plt.show()

#plt.plot((np.arange(len(gt_angle))+1), gt_angle, label="ground truth steering angle")
#plt.plot((np.arange(len(pred_angle))+1), pred_angle, label="predicted steering angle") 
#plt.xlabel("Sensor Frames") 
#plt.ylabel("Steering Angle Values")
#plt.legend() 
#plt.show() 

































