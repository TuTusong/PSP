import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img,mask_img
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from glob import glob
from PIL import Image


#############DATALOADER#######
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform  
        
        gt_dir = 'groundtruth'  
        lowlevel_dir = 'lowlevel'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, lowlevel_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x)  for x in clean_files if is_png_file(x)] 
        self.noisy_filenames = [os.path.join(rgb_dir, lowlevel_dir, x)       for x in noisy_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames) 
    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))  
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index]))) 
        clean_mask = mask_img(self.clean_filenames[tar_index])
        noisy_mask = mask_img(self.noisy_filenames[tar_index])
        weight_similar=np.zeros((64,64),np.uint8)
        for x in range(64):
          for y in range(64):
              if clean_mask[x,y] == noisy_mask[x,y]:
                 weight_similar[x,y]=255
              else :
                 weight_similar[x,y]=0
        weight = torch.from_numpy(np.float32(weight_similar))


        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]      

        return clean, noisy, clean_filename, noisy_filename,weight
    
#############DATALOADER_SELECT#######
class DataLoaderTrain_select(Dataset):
    def __init__(self, rgb_dir,img_options=None, target_transform=None):
        super(DataLoaderTrain_select, self).__init__()

        self.target_transform = target_transform  

        gt_dir = 'groundtruth'  
        lowlevel_dir = 'lowlevel'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir))) #获得clean文件，并进行排序
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, lowlevel_dir)))#获得lowlevel的文件，并进行排序
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x)  for x in clean_files if is_png_file(x)] 
        self.noisy_filenames = [os.path.join(rgb_dir, lowlevel_dir, x)       for x in noisy_files if is_png_file(x)]
        new_clean_filenames = []
        new_noisy_filenames = []
        for index in range(len(self.clean_filenames)):
            clean_mask = mask_img(self.clean_filenames[index])
            noisy_mask = mask_img(self.noisy_filenames[index])
            similar = 0
            all_num = 0
            for x in range(64):
                for y in range(64):
                    if clean_mask[x,y] == noisy_mask[x,y]:
                        similar += 1 
                    all_num +=1
            similar= similar/all_num
            if similar >= 0.90:
                new_clean_filenames.append(self.clean_filenames[index])
                new_noisy_filenames.append(self.noisy_filenames[index])
                # print(index)
                # print('--' +str(len(new_clean_filenames)))
        self.tar_size = len(new_clean_filenames) 
        self.clean_filenames = new_clean_filenames
        self.noisy_filenames = new_noisy_filenames
        
        self.img_options=img_options



    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))  
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index]))) 
        clean_mask = mask_img(self.clean_filenames[tar_index])
        noisy_mask = mask_img(self.noisy_filenames[tar_index])
        weight_similar=np.zeros((64,64),np.uint8) #mask size 64
        for x in range(64):
          for y in range(64):
              if clean_mask[x,y] == noisy_mask[x,y]:
                 weight_similar[x,y]=255
              else :
                 weight_similar[x,y]=0
        weight_similar= weight_similar.astype(np.float32)
        weight_similar = weight_similar/255
        weight = torch.from_numpy(np.float32(weight_similar))


        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]      

        return clean, noisy, clean_filename, noisy_filename,weight

    



##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        lowlevel_dir = 'lowlevel'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, lowlevel_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, lowlevel_dir, x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]


        return clean, noisy, clean_filename, noisy_filename


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)

def get_training_data_select(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain_select(rgb_dir, img_options,None)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)