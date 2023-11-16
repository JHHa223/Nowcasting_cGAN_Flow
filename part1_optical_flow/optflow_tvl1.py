# Calculating Optical Flow via TV-L1

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
import os

base_path = '../optflow_vis_test/'
path_list = os.listdir(base_path)

print(path_list)


for i in range(len(path_list)):

    if not os.path.isdir('../optflow_vis_test/'+path_list[i]+'/linear_o'):
        os.makedirs('../optflow_vis_test/'+path_list[i]+'/linear_o')

    
    new_path = '../optflow_vis_test/'+path_list[i]+'/linear_o/'
    
    path = glob.glob(base_path+path_list[i]+'/*.npy')
    path = sorted(path)
    print(path)
    
#### Single-Temporal Model ####


    for j in range(0,18):
        # Reading input data
        img0 = np.load(path[j]) # input image at t-10
        img1 = np.load(path[j+1]) # input image at t

        # Calculating optical flow field
        dtvl1=cv2.optflow.DualTVL1OpticalFlow_create()
        flow_1 = dtvl1.calc(img1, img0, None)

        # Saving optical flow field
        np.save(new_path+'o_vector_202007301000+'+str((j+1)*10)+'min.npy',flow_1)

        # Generating future frame using optical flow field
        h,w,_ = flow_1.shape
        flow_1[:,:,0] += np.arange(w)
        flow_1[:,:,1] += np.arange(h)[:,np.newaxis]
        Img_f = cv2.remap(img1, flow_1, None, cv2.INTER_CUBIC)
        
        #saving future frame at t+10
        np.save(new_path+'o_precipitation_202007301000+'+str(10*(j+1))+'min.npy',Img_f)

#### Multi-Temporal Model ####

    
    for j in range(0,18):
        # Reading input data
        img0 = np.load(path[j]) # input image at t-30
        img1 = np.load(path[j+1]) # input image at t-20
        img2 = np.load(path[j+2]) # input image at t-10
        img3 = np.load(path[j+3]) # input image at t

        # Calculating optical flow field
        dtvl1=cv2.optflow.DualTVL1OpticalFlow_create()
        flow_10 = dtvl1.calc(img1, img0, None)*1
        flow_20 = dtvl1.calc(img2, img1, None)*(1/2)
        flow_30 = dtvl1.calc(img3, img2, None)*(1/3)

        flow_tot = (flow_10+flow_20+flow_30)/3
        
        # Saving optical flow field
        np.save(new_path+'o_vector_Multi_202007301000+'+str((j+1)*10)+'min.npy',flow_tot)

        h,w,_ = flow_tot.shape
        flow_tot[:,:,0] += np.arange(w)
        flow_tot[:,:,1] += np.arange(h)[:,np.newaxis]
        Img_Multi_f = cv2.remap(img1, flow_tot, None, cv2.INTER_CUBIC)
        
        #saving future frame at t+10
        np.save(new_path+'o_precipitation_Multi_202007301000+'+str(10*(j+1))+'min.npy',Img_Multi_f)
