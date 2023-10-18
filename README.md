# Shadow Removal via Shadow Decomposition
This project aims to use deep learning method for shadow removal. Inspired by physical models of shadow formation, they use a linear illumination transformation to model the shadow effects in the image that allows the shadow image to be expressed as a combination of the shadow-free image, the shadow parameters, and a matte layer. The model uses two deep networks, namely SP-Net and M-Net, to predict the shadow parameters and the shadow matte respectively. 

![image](https://github.com/venkydesai/Shadow_removal_via_shadow_decomposition/assets/117113574/f85e754d-1cf0-4453-bcd5-c51495824c66)


* Drive Link - https://drive.google.com/drive/folders/19F7Q3jUY1k_FubXtcR1aimDrTSLtWpK7?usp=sharing

## Reference work :

> Le, H., & Samaras, D. (2019). Shadow removal via shadow image decomposition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 8578-8587).

## Instruction to project : 

The python file `train.py` under the `src/model` directory contains the training code. We use VGG16 and UNet-256 for our training over the images. 
To run inference use the `inference.py` file also under the similar dir using the command :
```
python3 inference.py
```
The `train.py` file also holds the `option()` function which can be used to change the model parameters.  


## Results :
![test](https://github.com/venkydesai/Shadow_removal_via_shadow_decomposition/assets/117113574/467e1120-37a4-43b7-9157-ed3e4c78f3b2)
