import os
from torch.utils.data import Dataset, random_split
from DatasetFromSubset import DatasetFromSubset
from TinyImageNet import TinyImageNet

extracted_folder = "/Users/srinivasang/code/eva4-2/week2/experiment/tiny-imagenet-200/"

def get_classes(download_folder):
    classes = []
    wnids = open(os.path.join(download_folder,"wnids.txt"), "r")
    for line in wnids:
        classes.append(line.strip())
    return classes
  
def TinyImageProcessor(train_split = 70,test_transforms = None,train_transforms = None):
  print("start")
  classes = get_classes("/Users/srinivasang/code/eva4-2/week2/experiment/tiny-imagenet-200/")
  dataset = TinyImageDataSet(classes,"/Users/srinivasang/code/eva4-2/week2/experiment/tiny-imagenet-200/")
  train_len = len(dataset)*train_split//100
  test_len = len(dataset) - train_len 
  train_set, val_set = random_split(dataset, [train_len, test_len])
  print(len(train_set))
  print(len(val_set))
  train_dataset = DatasetFromSubset(train_set, transform=train_transforms)
  test_dataset = DatasetFromSubset(val_set, transform=test_transforms)
  print(len(train_dataset))
  print(len(test_dataset))
  return train_dataset, test_dataset,classes

TinyImageNetDataSet()
    
