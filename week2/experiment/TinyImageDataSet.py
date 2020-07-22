import torch.utils.data
from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook

def class_names(url = "tiny-imagenet-200/wnids.txt"):
  f = open(url, "r")
  classes = []
  for line in f:
    classes.append(line.strip())
  return classes

class TinyImageDataSet(Dataset):
    def __init__(self,classes,url):
        self.data = []
        self.target = []
        self.classes = classes
        self.url = url
        print(url)
        wnids = open(url + "wnids.txt", "r")
        wclasses = wnids.read().splitlines()
        
        for wclass in wclasses:
          wclass = wclass.strip()
          for i in os.listdir(url+'/train/'+wclass+'/images/'):
            img = Image.open(url+"/train/"+wclass+"/images/"+i)
            print(url+"/train/"+wclass+"/images/"+i)
            npimg = np.asarray(img)
            
            if(len(npimg.shape) ==2):
             
               npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
            self.data.append(npimg)  
            self.target.append(self.classes.index(wclass))

        val_file = open(url + "/val/val_annotations.txt", "r")
        valfiles  = val_file.read().splitlines()
        for i in valfiles:
          split_img, split_class = i.strip().split("\t")[:2]
          img = Image.open( url + "/val/images/" + split_img)
          print(url + "/val/images/" +split_img)
          npimg = np.asarray(img)
          if(len(npimg.shape) ==2):
                    
                npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
          self.data.append(npimg)  
          self.target.append(self.classes.index(split_class))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        img = data     
        return data,target