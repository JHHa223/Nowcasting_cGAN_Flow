# Nowcasting_cGAN_Flow(Ha & Lee 2023)

## Introduction
The nowcasting model comprises optical flow algorithm and conditional GAN.

## Model Architecture 
A Deep Learning Model for Precipitation Nowcasting consists of two parts.

### Part I. Optical flow calculation

OpenCV library is used here to estimate the optical flow field.

The codes for optical flow calculation are provided in the directory "optical_flow".

#### Example: Optical flow estimation
```python
#### Single-Temporal Model ####

# Reading input data
img0 = np.load('path/img/radar_001.npy') # input image at t-10
img1 = np.load('path/img/radar_002.npy') # input image at t

# Calculating optical flow field
dtvl1=cv2.optflow.DualTVL1OpticalFlow_create()
flow_1 = dtvl1.calc(img1, img0, None)

# Saving optical flow field
np.save('o_vector_202007301000+10min.npy',flow_1)

# Generating future frame using optical flow field
h,w,_ = flow_1.shape
flow_1[:,:,0] += np.arange(w)
flow_1[:,:,1] += np.arange(h)[:,np.newaxis]
img_f = cv2.remap(img1, flow_1, None, cv2.INTER_CUBIC)
        
#saving future frame at t+10
np.save('o_precipitation_202007301000+10min.npy',img_f)
```

### Part II. conditional GAN architecture for training the nonlinear motion of precipitation fields


## Reference
Please refer the following publication for more details.

Ha, J.-H., & Lee, H. (2023). Enhancing Rainfall Nowcasting Using Generative Deep Learning Model with Multi-Temporal Optical Flow. Remote Sensing, 15(21), 5169; https://doi.org/10.3390/rs15215169
