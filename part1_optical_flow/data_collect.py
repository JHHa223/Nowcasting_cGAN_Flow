import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image, ImageFilter
import os

base_path = '/data0/jhha223/pix2pix/train_set/'
path_list = os.listdir(base_path)

for i in range(len(path_list)):

    path = glob.glob(base_path+path_list[i]+'/*.png')
    path = sorted(path)
 
    for j in range(0,len(path)-2,2):
        t1 = Image.open(path[j]).convert('RGB') # image produced by optical flow
        t2 = Image.open(path[j+1]).convert('RGB') # ground truth
        t1 = np.asarray(t1)
        t2 = np.asarray(t2)

        img = np.concatenate((t1, t2),axis=2)

        np.save('train_set/'+path_list[i]+'/img'+str(int(j/2))+'.npy',img)
