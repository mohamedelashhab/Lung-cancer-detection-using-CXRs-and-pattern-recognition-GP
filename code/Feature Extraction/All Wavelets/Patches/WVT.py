import os
import os.path
import random
import numpy as np
import pywt
import time
import cv2
from sklearn.feature_selection import VarianceThreshold
families = []

# gathering all the wavelet families into one list .
for family in pywt.families():
    families = families + pywt.wavelist(family)
levels  = [2]
image_type = ['ad','da','dd']

states = []
nodules = []
patches = []

'''

Get_Coordinates(file) 

parameters file :  str ,the path to the file that contain that contain clinical coordrmation 
about the data set.

the file name is "CLNDAT_EN.txt".

for example : Get_Coordinates ('F://another_folder/dataset/clinical/CLNDAT_EN.txt'). 

Returns  

coordinates :  dictionary  ,the key is image name ,it contain the coordinates of the tumor in the noduled image.

for example coordinates['JPCLN001.png'].

'''
def Get_Coordinates(file):
    coordinates = {} 
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

   # getting the x and y coordinates of the tumor 
    for i in content:
        nodule_X_cor = int(i.split('\t')[5])
        nodule_Y_cor = int(i.split('\t')[6])
        name = str(i.split('\t')[0][:-3])+'png'

        #mapping the coordinates with the image name 
        coordinates[name] = (nodule_X_cor,nodule_Y_cor)
    return coordinates

'''
 cut_nodules
    parameters path :  str ,the path to the data set that contain the noduled image.
               coord : dictionary , the key is the image name and it contains the x and y coordinates of the nodule.
 for example:
     cut_nodules('F://another_folder/dataset/',coord)

     it uses global states to append the shapes of the nodules image.

'''

def cut_nodules(path,coord):
    global nodules 
    global states
    for imgName in os.listdir(path):
        
        if imgName[-3:] == 'png' and imgName[3] == 'L':
           
           # reading the image and defining the nodule coordinates
            img = np.array(cv2.imread(path+'/{}'.format(imgName)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            coor = (coord[imgName][0] ,coord[imgName][1])

            #cutting a square shape with the nodule at the center
            origin = img[coor[1]-32:coor[1]+32,coor[0]-32:coor[0]+32]
            origin2 = np.array(origin)
            
            
            nodules.append(origin)
            states.append(1)
            num_rows, num_cols = origin.shape[:2]       
        
            # rotating the shape wiht 90 degrees each time and flipping each vertically and horizontally  
            rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -90, 1)
            for i in range(3):
                origin2 = cv2.warpAffine(origin2, rotation_matrix, (num_cols, num_rows))
                nodules.append(origin2)
                states.append(1)
            img=cv2.flip(origin,1)

            for i in range(3):
                img = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
                nodules.append(img)       
                states.append(1)


'''
cut_pathces(path)

prameters path : str, the path to the data set that conatin the images.

Description :
this function is used to cut the original image into patches each patch is a 64*64 matrix. 

     it uses global patches to append the shapes of the nodules image.

'''

def cut_pathces(path):
    global nodules
    global states
    global patches
    total_normal = []
    for imgName in os.listdir(path):
        if imgName[-3:] == 'png' and imgName[3] == 'N': 
            img = np.array(cv2.imread(path+'/{}'.format(imgName)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            width,height=0,0
            while height+64 <= img.shape[0]:
                width=0
                while width+64 <= img.shape[1]:
                    new = img[width:width+64,height:height+64]
                    width = width+64
                    if np.count_nonzero(new==0) == 0:
                        patches.append(new)
                height = height+64       
        
            
'''
 Wavelet_Features
 
 Parameters dset : list of (np.array) , images after converting them to np array .
            patches :list, contain the patches of all images.
            states : list , list that tells wheter the image noduled or not, contains 0's and 1's .
            mode : str ,the required wavelet family , if none it will be "db1"  .
            level : int , maximum level or scale .
            img_type : str ,one of the three types of returned image 'ad','da','dd' ,if none it will be 'ad' .
           

for example :
    Wavelet_features(images,states,'haar',4,'dd','C://Desktop') 

'''
def Wavelet_Features(dset,patches,states,mode='db1',level=6,image_type='ad'):
    new_dset = []
    new_normal = []
    for img in dset:
        try:
            # applying wavelet transformation with spicfic paramters
            Coefficients = pywt.wavedecn(img, mode, level=level)
            Result = Coefficients[1][image_type]

            # converting the feature from 2D matrix to 1D array
            feature = Result.ravel()
            new_dset.append(feature)
       
        except:
            continue
    for img in patches:
        try:

            Coefficients = pywt.wavedecn(img, mode, level=level)
            Result = Coefficients[1][image_type]

            feature = Result.ravel()
            new_normal.append(feature)
        except:
            continue
  
    try:
        # applying Feature selection
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        vector = sel.fit_transform(new_dset)
        vector = vector.tolist()
        normal_vector = sel.fit_transform(new_normal)
        normal_vector = normal_vector.tolist()
        normal_vector = normal_vector[0:len(vector)]
    except:
        with open('system4_failur.txt', 'a') as f:
            f.write('%s\n' % 'family__{}__level__{}__image_type__{}'.format(mode,level,image_type))
        return
    vector = vector + normal_vector
    states = states + 1078*[-1]
    label = ['a{}'.format(i) for i in range(1,len(vector[0])+1)]
    label = str(label)
    label = label.replace("'",'')
    label = label[1:-1]+',class'
    with open('E://clg/Dr waleed work/testing/family__{}__level__{}__image_type__{}.txt'.format(mode,level,image_type), 'a') as f:
        f.write('%s\n' % str(label))
    for i,state in zip(vector,states):
        i.append(state)
        #print(len(i))
        i = str(i)
        i = i[1:-1]
        with open('E://clg/Dr waleed work/testing/family__{}__level__{}__image_type__{}.txt'.format(mode,level,image_type), 'a') as f:
            f.write('%s\n' % str(i))
        

'''
just a main function we use to organize the whole process

'''
def main():
    global nodules
    global states
    global patches
    path = "E://clg/Graduation project/Data set/DS"
    file = 'E://clg/Graduation project/Data set/Clinical_Information/CLNDAT_EN.txt'
    coord = Get_Coordinates(file)
    cut_nodules(path,coord)
    cut_pathces(path)
    random.shuffle(patches)
  
    counter = 0
    for level in levels:
        for family in families:
            for imType in image_type:
                Wavelet_Features(nodules,patches,states,family,level,imType)
                counter = counter + 1
                print(counter)
    pass
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))