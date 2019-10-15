import os
import os.path
import random
import time

import numpy as np
import pywt
from sklearn.feature_selection import VarianceThreshold

import cv2

'''
Read_data(path) 

parameters path :  str ,the path to the data set that contain ".png" images . 

for example : Read_data ('F://another_folder/dataset')   

Returns  

Dset :  list ,  contain all the images in the data set folder .
state :  list ,  contain the state of the image whether the image noduled or non noduled,
it will contain 0's and 1's , 1 if the image is noduled and 0 otherwise .

'''

def Read_data(path):
    Dset = []
    state = []
    for f in os.listdir(path):
        if f[-3:] == 'png':
        
          if f[3] == 'L':    
              state.append(1)
        
          else:              
              state.append(0)
          
          # reading the image as np array
          img = np.array(cv2.imread(path+'/{}'.format(f)))
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          Dset.append(img)
        
    return Dset,state

'''
 Wavelet_Features
 
 Parameters dset : list of (np.array) , images after converting them to np array .
            states : list , list that tells wheter the image noduled or not, contains 0's and 1's .
            mode : str ,the required wavelet family , if none it will be "db1"  .
            level : int , maximum level or scale .
            img_type : str ,one of the three types of returned image 'ad','da','dd' ,if none it will be 'ad' .
            path_folder : str, the path to the folder in which the features will be written.

for example :
    Wavelet_features(images,states,'haar',4,'dd','C://Desktop') 

'''
def Wavelet_Features(dset,states,mode='db1',level=6,img_type='ad',path_folder="F://ashhab/system2"):
    new_dset = []
     
    for img in dset:
        try:
           
           # applying wavelet transformation with spicfic paramters
            Coefficients = pywt.wavedecn(img, mode, level=level)
            Result = Coefficients[1][img_type]
            
            # converting the feature from 2D matrix to 1D array
            feature = Result.ravel()
            
            new_dset.append(feature)
        except:
            continue
    try:

        # applying Feature selection
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        vector = sel.fit_transform(new_dset)
        new_dset = vector.tolist()

    except:
        with open('system2_failur.txt', 'a') as f:
            f.write('%s\n' % 'family__{}__level__{}__image_type__{}'.format(mode,level,img_type))
        return

    label = ['a{}'.format(i) for i in range(1, len(new_dset[0])+1)]
    label = str(label)
    label = label.replace("'",'')
    label = label[1:-1]+',class'

    # Writing the feature to the specified path (parameter path_folder).

    with open('{}/family__{}__level__{}__image_type__{}.txt'.format(path_folder,mode,level,img_type), 'a') as f:
        f.write('%s\n' % str(label))
    for i,state in zip(new_dset,states):
        i.append(state)
        i = str(i)
        i = i[1:-1]
        with open('{}/family__{}__level__{}__image_type__{}.txt'.format(path_folder,mode,level,img_type), 'a') as f:
            f.write('%s\n' % str(i))
        



'''
just a main function we use to organize the whole process

'''
def main():

    levels = [4]
    img_type = ['ad','da','dd']
    families = []
    for family in pywt.families():
        families = families + pywt.wavelist(family)



    pathToDataset = "E://clg/Graduation project/Data set/DS"
    pathToFolder = "E://clg/Dr waleed work/testing"
    images,states = Read_data(pathToDataset)
    counter = 0
    for level in levels:
        for mode in families:
            for img in img_type:
                Wavelet_Features(images,states,mode,level,img,pathToFolder)
                counter = counter + 1
                print(counter)
    
    pass

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
