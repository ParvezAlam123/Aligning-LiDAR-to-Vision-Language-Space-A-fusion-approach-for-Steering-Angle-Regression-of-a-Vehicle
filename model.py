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
import matplotlib.pyplot as plt 




device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 





class Network(nn.Module):
    def __init__(self):
        super().__init__() 
        self.flave_model = FlavaModel.from_pretrained("facebook/flava-full")
        self.processor = AutoProcessor.from_pretrained("facebook/flava-full") 

        for param in self.flave_model.parameters():
           param.requires_grad = False

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=768, kernel_size=3)

    def forward(self, voxels, image, annotation): 
        voxels = voxels.permute(0, 3, 1, 2)
        x = F.gelu(self.conv1(voxels))  
        #print("conv1", x.shape)
        x = F.gelu(self.conv2(x)) 
        #print("conv2", x.shape)
        x = F.gelu(self.conv3(x)) 
        #print("conv3", x.shape)
        x = F.gelu(self.conv4(x)) 
        #print("conv4", x.shape)
        x = self.conv5(x).permute(0, 2, 3, 1)
        out = torch.max(x, dim=2)[0]

        inputs = self.processor(text=annotation, images=image[0], return_tensors="pt", padding=True).to(device)
        outputs = self.flave_model(**inputs)
        image_embeddings = outputs.image_embeddings 
        text_embeddings = outputs.text_embeddings 
        #print(image_embeddings.shape)
        #print(text_embeddings.shape) 
        #print(out.shape)
        return out, image_embeddings,  text_embeddings
        
       


class Attention(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.fc2 = nn.Linear(2*dim, dim)
        
        #self.bn = nn.BatchNorm2d(64) 

        
        
    def forward(self, current_feature, past_feature): 
        B, C, dim = current_feature.shape 
        
        scale = np.sqrt(2 * dim)
        
        Query = self.fc1(current_feature)
        Key = Value = self.fc1(past_feature)
        
        weight = (Query @ Key.transpose(1,2)) / scale 
        weight = F.softmax(weight, dim=-1)  
        
        weight_numpy = weight[0].detach().cpu().numpy() 
        print("weight = ", weight_numpy.shape) 


        # Create a heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(weight_numpy, cmap='viridis', aspect='auto')

        # Add a color bar
        plt.colorbar(label='Attention Weights')

        # Add labels for rows and columns
        plt.xlabel('LiDAR Features-Image Features')
        plt.ylabel('Image Featuires- LiDAR Features')

        # Annotate each cell with the weight value
        #for i in range(attention_weights.shape[0]):
        #    for j in range(attention_weights.shape[1]):
        #        plt.text(j, i, f'{attention_weights[i, j]:.2f}', ha='center', va='center', color='white')

        plt.title('Attention Weights Heatmap')
        plt.show()

        attention_feature = weight @ Value  

        feature = Query + attention_feature          # residual connection 

        feature = F.relu(self.fc2(feature))
        #print(feature.shape)
        #feature = self.bn(feature)

        return feature 



class AngleNetwork(nn.Module):
    def __init__(self):
        super().__init__() 

        self.model_backbone = Network() 
        self.model_backbone.load_state_dict(torch.load("model_state_alignment.pth"))
        self.model_backbone.to(device)
        self.model_backbone.eval() 

        for param in self.model_backbone.parameters():
           param.requires_grad = False 

        self.attention_block = Attention() 


        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, voxels, image):
        annotation  = ["This is the driving scene."]
        out, image_embeddings,  text_embeddings =  self.model_backbone(voxels, image, annotation)  
        print("lidar shape = ", out.shape, "img shape = ", image_embeddings.shape)
        x = torch.cat((out, image_embeddings), dim=1) 
        # uncomment for lidar only feature 
        #x = image_embeddings 
        x = self.attention_block(x, x)
        x = torch.max(x, dim=1)[0]
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = self.fc4(x)[0]
        return x  
    
    
    








