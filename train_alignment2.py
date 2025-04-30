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
from dataset import Data 
from model import Network  
import matplotlib.pyplot as plt 







device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 


pcd_path = "/media/parvez/Expansion/TiHAN_maini/maingate_testbed/pcd" 
camera_path = "/media/parvez/Expansion/TiHAN_maini/maingate_testbed/Camera" 
json_path = "/media/parvez/Expansion/TiHAN_maini/maingate_testbed/Annotions.json" 


train_dataset = Data(pcd_path=pcd_path, camera_path=camera_path, json_path=json_path)
train_loader = DataLoader(dataset = train_dataset, batch_size=1, shuffle=False) 


model = Network() 
model.to(device) 


optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) 

training_loss = [] 


def train(model, epochs):
    for i in  range(epochs):
       running_loss = 0.0 
       for n, data in enumerate(train_loader):
          voxels = data["voxels"].float().to(device) 
          N, V, _, _ = voxels.shape
          if V == 0:
              continue
          else:
             image = data["image"].float().to(device)
             annotation = data["annotation"] 
             voxel_embeddings, image_embeddings, text_embeddings = model(voxels, image, annotation) 
             voxel_vector = torch.max(voxel_embeddings, dim=1)[0] 
             image_vector = torch.max(image_embeddings, dim=1)[0]
             text_vector = torch.max(text_embeddings, dim=1)[0]

             cosine_text_voxel = (text_vector * voxel_vector).sum() / (torch.linalg.norm(text_vector) * torch.linalg.norm(voxel_vector)) 

             cosine_text_image = (text_vector * image_vector).sum() / (torch.linalg.norm(text_vector) * torch.linalg.norm(image_vector)) 

             loss = (1 - cosine_text_voxel) + (1 - cosine_text_image) 
           
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
       torch.save(checkpoint, "same_embedding_space.pth")
        




train(model, 60) 


plt.plot(np.arange(60), training_loss)
plt.show()