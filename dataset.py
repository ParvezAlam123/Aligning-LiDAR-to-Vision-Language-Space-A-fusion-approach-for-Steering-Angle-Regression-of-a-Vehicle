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



@njit
def voxelize(points , voxel_size, grid_range, max_points_in_voxel = 60, max_num_voxels = 20000):
    points_copy = points.copy()
    grid_size = np.floor((grid_range[3:] - grid_range[:3]) / voxel_size).astype(np.int32)

    coor_to_voxelidx = np.full((grid_size[2], grid_size[1], grid_size[0]), -1, dtype=np.int32)
    voxels = np.zeros((max_num_voxels, max_points_in_voxel, points_copy.shape[-1]), dtype=points_copy.dtype)
    coors = np.zeros((max_num_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(shape=(max_num_voxels,), dtype=np.int32)

    coor = np.floor((points_copy[:, :3] - grid_range[:3]) / voxel_size).astype(np.int32)
    mask = np.logical_and(np.logical_and((coor[:, 0] >= 0) & (coor[:, 0] < grid_size[0]),
                                         (coor[:, 1] >= 0) & (coor[:, 1] < grid_size[1])),
                          (coor[:, 2] >= 0) & (coor[:, 2] < grid_size[2]))
    coor = coor[mask, ::-1]
    points_copy = points_copy[mask]
    assert points_copy.shape[0] == coor.shape[0]

    voxel_num = 0
    for i, c in enumerate(coor):
        voxel_id = coor_to_voxelidx[c[0], c[1], c[2]]
        if voxel_id == -1:
            voxel_id = voxel_num
            if voxel_num > max_num_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[c[0], c[1], c[2]] = voxel_id
            coors[voxel_id] = c
        n_pts = num_points_per_voxel[voxel_id]
        if n_pts < max_points_in_voxel:
            voxels[voxel_id, n_pts] = points_copy[i]
            num_points_per_voxel[voxel_id] += 1

    return voxels[:voxel_num], coors[:voxel_num], num_points_per_voxel[:voxel_num]   




class Data(Dataset):
    def __init__(self, pcd_path, camera_path, json_path):
        self.pcd_path = pcd_path 
        self.camera_path = camera_path 
        self.json_path = json_path  

        with open(self.json_path, "r") as f:
            annotation_data = json.load(f)

        pcd_files = sorted(os.listdir(self.pcd_path))
        camera_files = sorted(os.listdir(self.camera_path))  
        self.files = [] 
        
        i = 0 
        j = 0 
        while(i < len(pcd_files) and j < len(camera_files)): 
            sample = {} 
            pcd_frame_path = os.path.join(self.pcd_path, pcd_files[i]) 
            camera_frame_path = os.path.join(self.camera_path, camera_files[j]) 
            annotation = annotation_data[camera_files[j]] 
            sample["pcd_frame"] = pcd_frame_path 
            sample["camera_frame"] = camera_frame_path 
            sample["annotation"] = annotation 
            self.files.append(sample)

            if (i+1 < len(pcd_files) and j+1 <len(camera_files)):
               sampel = {} 
               pcd_frame_path = os.path.join(self.pcd_path, pcd_files[i+1]) 
               camera_frame_path = os.path.join(self.camera_path, camera_files[j+1]) 
               annotation = annotation_data[camera_files[j+1]]
               sample["pcd_frame"] = pcd_frame_path 
               sample["camera_frame"] = camera_frame_path 
               sample["annotation"] = annotation 
               self.files.append(sample)

            i = i + 2 
            j = j + 3  


    def __len__(self):
        return len(self.files) 


    def __getitem__(self, index):
       pcd_frame_path = self.files[index]["pcd_frame"]
       camera_frame_path = self.files[index]["camera_frame"]
       annotation = self.files[index]["annotation"]

       #print(pcd_frame_path)
       pcd = o3d.io.read_point_cloud(pcd_frame_path)
       points = np.asarray(pcd.points)  
     
       voxels, coors, num_points_per_vexel = voxelize(points=points, voxel_size=np.array([0.2, 0.2, 0.4]), grid_range=np.array([0, -40, -3, 70.4, 40, 1])) 
       image = cv2.imread(camera_frame_path) 
        
       return {"voxels":voxels, "image":image, "annotation":annotation}
         





class AngleData(Dataset):
    def __init__(self, pcd_path, camera_path, angle_path, train=True):
        self.pcd_path = pcd_path  
        self.camera_path = camera_path 
        self.angle_path = angle_path  
        self.train = train

        with open(self.angle_path, "r") as f:
            data = json.load(f)
        camera_files = sorted(os.listdir(self.camera_path)) 
        pcd_files = sorted(os.listdir(self.pcd_path)) 
        print("cemare", len(camera_files), "pcd_files", len(pcd_files))
        self.files = [] 
        #if self.train: 
        i = 0 
        j = 0 
        k = 0
        if train:
             while(i < 7000 and j < len(camera_files)) and k < len(data): 
                sample = {} 
                pcd_frame_path = os.path.join(self.pcd_path, pcd_files[i]) 
                camera_frame_path = os.path.join(self.camera_path, camera_files[j]) 
                angle = data[k]["data"]
                sample["pcd_frame"] = pcd_frame_path 
                sample["camera_frame"] = camera_frame_path 
                sample["angle"] = angle
                self.files.append(sample)

                if (i+1 < len(pcd_files) and j+1 <len(camera_files)):
                   sampel = {} 
                   pcd_frame_path = os.path.join(self.pcd_path, pcd_files[i+1]) 
                   camera_frame_path = os.path.join(self.camera_path, camera_files[j+1]) 
                   angle = data[k+4]["data"]
                   sample["pcd_frame"] = pcd_frame_path 
                   sample["camera_frame"] = camera_frame_path 
                   sample["angle"] = angle 
                   self.files.append(sample)

                i = i + 2 
                j = j + 3  
                k = k + 8 
        else:
            while(7000+i < len(pcd_files) and 10500+j < len(camera_files)) and 28000+k < len(data): 
                sample = {} 
                pcd_frame_path = os.path.join(self.pcd_path, pcd_files[7000+i]) 
                camera_frame_path = os.path.join(self.camera_path, camera_files[10500+j]) 
                angle = data[28000+k]["data"]
                sample["pcd_frame"] = pcd_frame_path 
                sample["camera_frame"] = camera_frame_path 
                sample["angle"] = angle
                self.files.append(sample)

                if (7000+i+1 < len(pcd_files) and 10500+j+1 <len(camera_files)):
                   sampel = {} 
                   pcd_frame_path = os.path.join(self.pcd_path, pcd_files[7000+i+1]) 
                   camera_frame_path = os.path.join(self.camera_path, camera_files[10500+j+1]) 
                   angle = data[28000+k+4]["data"]
                   sample["pcd_frame"] = pcd_frame_path 
                   sample["camera_frame"] = camera_frame_path 
                   sample["angle"] = angle 
                   self.files.append(sample)

                i = i + 2 
                j = j + 3  
                k = k + 8 

    def __len__(self):
        return len(self.files) 


    def __getitem__(self, index):
       pcd_frame_path = self.files[index]["pcd_frame"]
       camera_frame_path = self.files[index]["camera_frame"]
       angle = self.files[index]["angle"]

       #print(pcd_frame_path)
       pcd = o3d.io.read_point_cloud(pcd_frame_path)
       points = np.asarray(pcd.points) 

       voxels, coors, num_points_per_vexel = voxelize(points=points, voxel_size=np.array([0.2, 0.2, 0.4]), grid_range=np.array([0, -40, -3, 70.4, 40, 1])) 
       image = cv2.imread(camera_frame_path)  


       return {"voxels":voxels, "image":image, "angle":angle}
    




    
         


class PCDDATATRAIN(Dataset):
    def __init__(self, pcd_path, camera_path, angle_path, train=True):
        self.pcd_path = pcd_path  
        self.camera_path = camera_path 
        self.angle_path = angle_path  
        self.train = train

        with open(self.angle_path, "r") as f:
            data = json.load(f)
        camera_files = sorted(os.listdir(self.camera_path)) 
        pcd_files = sorted(os.listdir(self.pcd_path)) 
        #print("cemare", len(camera_files), "pcd_files", len(pcd_files))
        self.files = [] 
        #if self.train: 
        i = 0 
        j = 0 
        k = 0
        if train:
             while(i < 7000 and j < len(camera_files)) and k < len(data): 
                sample = {} 
                pcd_frame_path = os.path.join(self.pcd_path, pcd_files[i]) 
                camera_frame_path = os.path.join(self.camera_path, camera_files[j]) 
                angle = data[k]["data"]
                sample["pcd_frame"] = pcd_frame_path 
                sample["camera_frame"] = camera_frame_path 
                sample["angle"] = angle
                self.files.append(sample)

                if (i+1 < len(pcd_files) and j+1 <len(camera_files)):
                   sampel = {} 
                   pcd_frame_path = os.path.join(self.pcd_path, pcd_files[i+1]) 
                   camera_frame_path = os.path.join(self.camera_path, camera_files[j+1]) 
                   angle = data[k+4]["data"]
                   sample["pcd_frame"] = pcd_frame_path 
                   sample["camera_frame"] = camera_frame_path 
                   sample["angle"] = angle 
                   self.files.append(sample)

                i = i + 2 
                j = j + 3  
                k = k + 8 
        else:
            while(7000+i < len(pcd_files) and 10500+j < len(camera_files)) and 28000+k < len(data): 
                sample = {} 
                pcd_frame_path = os.path.join(self.pcd_path, pcd_files[7000+i]) 
                camera_frame_path = os.path.join(self.camera_path, camera_files[10500+j]) 
                angle = data[28000+k]["data"]
                sample["pcd_frame"] = pcd_frame_path 
                sample["camera_frame"] = camera_frame_path 
                sample["angle"] = angle
                self.files.append(sample)

                if (7000+i+1 < len(pcd_files) and 10500+j+1 <len(camera_files)):
                   sampel = {} 
                   pcd_frame_path = os.path.join(self.pcd_path, pcd_files[7000+i+1]) 
                   camera_frame_path = os.path.join(self.camera_path, camera_files[10500+j+1]) 
                   angle = data[28000+k+4]["data"]
                   sample["pcd_frame"] = pcd_frame_path 
                   sample["camera_frame"] = camera_frame_path 
                   sample["angle"] = angle 
                   self.files.append(sample)

                i = i + 2 
                j = j + 3  
                k = k + 8 

        
            
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self,index): 

        pcd_frame_path = self.files[index]["pcd_frame"]
        camera_frame_path = self.files[index]["camera_frame"]
        angle = self.files[index]["angle"]

        #print(pcd_frame_path)
        pcd = o3d.io.read_point_cloud(pcd_frame_path)
        pcd = np.asarray(pcd.points) 

        #voxels, coors, num_points_per_vexel = voxelize(points=points, voxel_size=np.array([0.2, 0.2, 0.4]), grid_range=np.array([0, -40, -3, 70.4, 40, 1])) 
        #image = cv2.imread(camera_frame_path)  


        f_range = (0, 40)
        side_range = (10, -10)
        height_range=(-10, 10)
        
        #pcd_file = self.files[index]['pcd']
        
        #pcd = np.asarray(o3d.io.read_point_cloud(os.path.join(path, 'training/SteeringData', pcd_file)).points)
        
        point_x = pcd[:, 0]
        point_y = pcd[:, 1]
        point_z = pcd[:, 2]
        
        # projection on XY plane
        
        f_filt = np.logical_and( (point_x > f_range[0]), (point_x < f_range[1]))
        s_filt = np.logical_and((point_y < side_range[0]), (point_y > side_range[1]))
        indices = np.logical_and(f_filt, s_filt)


        point_xg = np.floor(point_x[indices])
        point_yg = np.floor(point_y[indices])
        point_zg = point_z[indices]

        im = np.zeros((20, 40))
        mean = np.zeros((20, 40))

        for i in range(len(point_xg)):
                row = int(point_yg[i])
                column = int(point_xg[i])
                im[row][column] = mean[row][column] + (point_zg[i] - mean[row][column]) / (i+1)

       
        
        pcd1 = torch.from_numpy(im)
        
        # projection on YZ plane
        
        h_filt = np.logical_and((point_z > height_range[0]), (point_z < height_range[1]))
        
        indices = np.logical_and(s_filt, h_filt)
        
        point_xh = np.floor(point_x[indices])
        point_yh = np.floor(point_y[indices])
        point_zh = np.floor(point_z[indices])
        
        im = np.zeros((20, 20))
        mean = np.zeros((20, 20))
        
        for i in range(len(point_yh)):
            row = int(point_zh[i])
            column = int(point_yh[i])
            im[row][column] = mean[row][column] + (point_xh[i] - mean[row][column]) / (i+1)
            
        
        pcd2 = torch.from_numpy(im)
        
        
        #angle = self.files[index]['gt']
        
        return {'pcd1':pcd1,'pcd2':pcd2, 'angle':float(angle)}  
    


    

            


class AngleDataNight(Dataset):
    def __init__(self, pcd_path, camera_path, angle_path):
        self.pcd_path = pcd_path  
        self.camera_path = camera_path 
        self.angle_path = angle_path  

        with open(self.angle_path, "r") as f:
            data = json.load(f)
        camera_files = sorted(os.listdir(self.camera_path)) 
        pcd_files = sorted(os.listdir(self.pcd_path)) 
        print("cemare", len(camera_files), "pcd_files", len(pcd_files))
        self.files = [] 
        #if self.train: 
        i = 0 
        j = 0 
        k = 0
        while(i < len(pcd_files) and j < len(camera_files)) and k < len(data): 
                sample = {} 
                pcd_frame_path = os.path.join(self.pcd_path, pcd_files[i]) 
                camera_frame_path = os.path.join(self.camera_path, camera_files[j]) 
                angle = data[k]["data"]
                sample["pcd_frame"] = pcd_frame_path 
                sample["camera_frame"] = camera_frame_path 
                sample["angle"] = angle
                self.files.append(sample)

                if (k+1 < len(data)):
                   sampel = {} 
                   pcd_frame_path = os.path.join(self.pcd_path, pcd_files[i]) 
                   camera_frame_path = os.path.join(self.camera_path, camera_files[j]) 
                   angle = data[k]["data"]
                   sample["pcd_frame"] = pcd_frame_path 
                   sample["camera_frame"] = camera_frame_path 
                   sample["angle"] = angle 
                   self.files.append(sample)
                
                if k%2==0:
                    i = i + 1 
                    j = j + 2 
                    k = k + 1 
                else:
                    i = i + 1 
                    j = j + 3 
                    k = k + 1 

        
    def __len__(self):
        return len(self.files) 


    def __getitem__(self, index):
       pcd_frame_path = self.files[index]["pcd_frame"]
       camera_frame_path = self.files[index]["camera_frame"]
       angle = self.files[index]["angle"]

       #print(pcd_frame_path)
       pcd = o3d.io.read_point_cloud(pcd_frame_path)
       points = np.asarray(pcd.points) 

       voxels, coors, num_points_per_vexel = voxelize(points=points, voxel_size=np.array([0.2, 0.2, 0.4]), grid_range=np.array([0, -40, -3, 70.4, 40, 1])) 
       image = cv2.imread(camera_frame_path)  


       return {"voxels":voxels, "image":image, "angle":angle}
    






