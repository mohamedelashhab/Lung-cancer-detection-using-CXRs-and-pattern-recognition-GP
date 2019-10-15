import os
import os.path
import random
import numpy as np
import pywt
import time
import cv2
from sklearn.feature_selection import VarianceThreshold
families = []
for family in pywt.families():
    families = families + pywt.wavelist(family)
levels  = [1]
image_type = ['ad']

states = []
nodules = []
normal = []
total_normal = []
def Read_info(path,file):
    info = {} 
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for i in content:
        width = int(i.split('\t')[5])
        height = int(i.split('\t')[6])
        name = str(i.split('\t')[0][:-3])+'png'
        info[name] = (width,height)
    return info


def cut_nodules(path,info):
    global nodules 
    global states
    for imgName in os.listdir(path):
        if imgName[-3:] == 'png' and imgName[3] == 'L':
            img = np.array(cv2.imread(path+'/{}'.format(imgName)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            coor = (info[imgName][0] ,info[imgName][1])
            origin = img[coor[1]-32:coor[1]+32,coor[0]-32:coor[0]+32]
            origin2 = np.array(origin)
            #cv2.imshow('origin',origin)
            #cv2.waitKey(0)
            nodules.append(origin)
            states.append(1)
            num_rows, num_cols = origin.shape[:2]       
            rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -90, 1)
            for i in range(3):
                origin2 = cv2.warpAffine(origin2, rotation_matrix, (num_cols, num_rows))
                #cv2.imshow('i',origin2)
                #cv2.waitKey(0)
                nodules.append(origin2)
                states.append(1)
            img=cv2.flip(origin,1)
            for i in range(3):
                img = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
                #cv2.imshow('i2',img)
                #cv2.waitKey(0)
                nodules.append(img)       
                states.append(1)


def patches(path,num):
    global nodules
    global states
    global normal
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
                    zeroz = np.count_nonzero(new==0)
                    nonzeroz = np.count_nonzero(new!=0)
                    shape = new.shape
                    percent = int((nonzeroz/(shape[0]*shape[1]))*100)
                    if percent >= num:
                        #cv2.imshow('new',new)
                        #cv2.waitKey(0)
                        normal.append(new)
                height = height+64       
           
        
            

def WVT(dset,normal,states,mode='db1',level=6,image_type='ad'):
    new_dset = []
    new_normal = []
    for img in dset:
        try:
            Coefficients = pywt.wavedecn(img, mode, level=level)
            Result = Coefficients[1][image_type]
            feature = Result.ravel()
            new_dset.append(feature)
        except:
            continue
    for img in normal:
        try:
            Coefficients = pywt.wavedecn(img, mode, level=level)
            Result = Coefficients[1][image_type]

            feature = Result.ravel()
            new_normal.append(feature)
        except:
            continue
    #print(len(new_dset))
    try:
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        vector = sel.fit_transform(new_dset)
        vector = vector.tolist()
        normal_vector = sel.fit_transform(new_normal)
        normal_vector = normal_vector.tolist()
        normal_vector = normal_vector[0:10240]
    except:
        with open('fail.txt', 'a') as f:
            f.write('%s\n' % 'family__{}__level__{}__image_type__{}'.format(mode,level,image_type))
        return
    #print(str(len(vector)),str(len(normal_vector)))
    vector = vector + normal_vector
    states = states + 10240*[-1]
    print(len(vector),len(states))
    label = ['a{}'.format(i) for i in range(1,len(vector[0])+1)]
    label = str(label)
    label = label.replace("'",'')
    label = label[1:-1]+',class'
    with open('F://GP/Project/Calculate FROC/Average FROC/family__{}__level__{}__image_type__{}.txt'.format(mode,level,image_type), 'w') as f:
        f.write('%s\n' % str(label))
    for i,state in zip(vector,states):
        i.append(state)
        #print(len(i))
        i = str(i)
        i = i[1:-1]
        with open('F://GP/Project/Calculate FROC/Average FROC/family__{}__level__{}__image_type__{}.txt'.format(mode,level,image_type), 'a') as f:
            f.write('%s\n' % str(i))
        

def main():
    global nodules
    global states
    global normal
    path = 'F://sdataset'
    file = 'F://GP/Clinical_Information/CLNDAT_EN.txt'
    info = Read_info(path,file)
    cut_nodules(path,info)

    #print(len(normal))
    #print("len of images" + str(len(nodules)),"  len of states" + str(len(states)))
    
    counter = 0
    for level in [1]:
        for family in ['bior3.9']:
            for imType in ['ad']:
                for per in [70]:
                    patches(path,per)
                    random.shuffle(normal)
                    WVT(nodules,normal,states,family,level,imType)
                    normal = []
                    counter = counter + 1
                    print(counter)
    pass
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
