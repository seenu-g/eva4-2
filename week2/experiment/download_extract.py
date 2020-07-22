import os
import zipfile
from io import StringIO,BytesIO
        
def extract_zipfile(download_folder,fileName):
    
    with zipfile.ZipFile(fileName, 'r') as zip: 
         try:
              #zip.printdir() 
              print('Extracting file ' + fileName + ' now... to ',download_folder) 
              zip.extractall() 
              print('File Extraction Done!') 
              return 
         except:
              print ("Exception to Unzip file",fileName)
              return 
          
IMAGES_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
fileName = "tiny-imagenet-200.zip"
download_folder = "/Users/srinivasang/code/eva4-2/week2/experiment/"

extract_zipfile(download_folder,fileName)

