# Nowcasting_cGAN_Flow

## Introduction
The nowcasting model comprises optical flow algorithm and conditional GAN (Ha & Lee Remote Sens. 2023, 15(21), 5169).

## Model Architecture 
A Deep Learning Model for Precipitation Nowcasting consists of two parts (i.e., linear extrapolation and conditional GAN).

### Part I. Linear extrapolation using optical flow field

OpenCV library is used here to estimate the optical flow field.

The codes for optical flow calculation are provided in the directory "part1_optical_flow".

#### Example: Optical flow estimation
```python
#### Single-Temporal Model ####

# Reading input radar images
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
We employed Pix2pix (Isola et al. 2017) to refine the nowcasting outputs produced by optical flow algorithm.

The codes are provided in the directory "part2_cGAN".

#### nowcasting_opt_gan.py
```
# Import models and dataset loader
from models import *
from dataset_loader import *

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

#Training dataset
dataloader = DataLoader(
    ImageDataset(is_train=0, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=True
)

#Validation dataset
val_dataloader = DataLoader(
    ImageDataset(is_train=1, transforms_=transforms_, mode="val"),
    batch_size=10,
    shuffle=True,
    num_workers=1,
    pin_memory=True
)
```


## Reference
Please refer the following publication for more details.

Ha, J.-H., & Lee, H. (2023). Enhancing Rainfall Nowcasting Using Generative Deep Learning Model with Multi-Temporal Optical Flow. Remote Sensing, 15(21), 5169; https://doi.org/10.3390/rs15215169.

Isola, P., Zhu, J. -Y., Zhou T. & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, pp. 5967-5976, https://doi.org/10.1109/CVPR.2017.632.
