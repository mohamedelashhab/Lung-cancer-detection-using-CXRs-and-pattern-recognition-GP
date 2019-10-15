import os
import os.path

import numpy as np
import pywt

import cv2
import matplotlib.path as mplPath


def convert(string):
    try:
        a,b = string[:-1].split(' ')
    except:
        if len(string[:-1].split(' ') )== 4:
            x,a,b,y = string[:-1].split(' ')
        else:
            s,x,y,a,b,z = string[:-1].split(' ')
    return float(a),float(b)

    
def process_L(path_L):
    left = {}
   
    for f in os.listdir(path_L[:-1]):
        new = []
        if f[-3:] == 'txt':
            with open(path_L+f) as fi:
                content = fi.readlines()
            content = list(map(convert,content[2:-1]))
            for i in content:
                i = list(i)
                i.reverse()
                new.append(i)
            left['{}png'.format(f[:-3])] = np.array(new)
            del new
    return left


def process_R(path_R):
    left = {}
    
    for f in os.listdir(path_R[:-1]):
        new = []
        if f[-3:] == 'txt':
            with open(path_R+f) as fi:
                content = fi.readlines()
            #print(content[2:-2])[0]
            content = list(map(convert,content[2:-1]))
            for i in content:
                i = list(i)
                i.reverse()
                new.append(i)
            left['{}png'.format(f[:-3])] = np.array(new)
            del new
    return left

def run(points_L,points_R,path_to_images,path_to_segmentation_folder):
    for img in os.listdir(path_to_images[:-1]):
        if img[-3:] == 'png':
            a = points_L[img]
            b = points_R[img]
            newImage = cv2.imread(path_to_images+img)
            newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
            bbPath_L = mplPath.Path(a*2)
            bbPath_R = mplPath.Path(b*2)
            for x in range(0,2048):
                for y in range(0,2048):
                    if (bbPath_L.contains_point((x, y))!=1 and bbPath_R.contains_point((x, y))!=1):
                        newImage[x][y]=255;
            cv2.imwrite(path_to_segmentation_folder+img,newImage)

            



def main():
    
    path_to_images = '/run/media/a4hab/1ECA933908590CB5/dataset/'
    path_to_save_segmentation = '/run/media/a4hab/1ECA933908590CB5/sdataset/'
    path_L = '/run/media/a4hab/1ECA933908590CB5/L/'
    path_R = '/run/media/a4hab/1ECA933908590CB5/R/'
    run(process_L(path_L),process_R(path_R),path_to_images,path_to_save_segmentation)
    pass

if __name__ == '__main__':
    main()


