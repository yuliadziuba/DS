# Summary
This repo details my solution to [Kaggle's Airbus Ship Detection](https://www.kaggle.com/competitions/airbus-ship-detection) challenge. The aim of the competition is to analyze satellite images, detect the ships 
and produce segmentation masks of the ships. We have original images (around 100 k of them) and corresponding masks for each occuring ship in .csv format. Also this repo includes a pipeline for training UNet model for 
the problem of ships detection. I will use notebook **run.ipynb** as main file in EDA below, all images from output **run.ipynb**. 

# Repository content
This repository consists of:
```

 ├── utils
     ├── loss.py
     ├── utils.py
     ├── generator.py
     ├── predict.py
 ├── run.ipynb 
 ├── config.py
 ├── train.py         
 ├── test.py 
 ├── requirements.txt
 ├── data
      ├── train_v2
      ├── test_v2
      ├── train_ship_segmentations_v2.csv
      ├── sample_submission_v2.csv
```
Firstly, we need to create a base directory (in my case it's named **'data'**), download dataset from kaggle: [dataset](https://www.kaggle.com/competitions/airbus-ship-detection/data) and unzip it in that **data** folder.  
The dataset has satellite images of ships in ocean or on docks as input images. The output are given in **train_segmentations_v2.csv** and **sample_submission_v2.csv** for training and testing respectively.
Those data are in the form of ImageId -> RLE Encoded Vector. The output of the images in dataset are encoded using Run Length Encoding (RLE), which based on [script](https://www.kaggle.com/code/paulorzp/run-length-encode-and-decode/script). Also the folders with images you can find in **train_v2** and **test_v2**

## Files description

 1. utils/loss.py file contains all losses (also custom metrics) that be used for this task.
   I prefer dice coefficients and loss function for this task.

 2. utils/utils.py is encoders and decoders, data visualization and masks as image file.
    
 3. utils/generator.py contains batch generator and data augmentation function.
 
 4. utils/predict.py includes predict functions.
 
 5. config.py consists of hyperparameter optimizations such as batch size, data scaling, epoch or number optimizer-steps per epoch. 
 
 6. train.py here we will train our model. test.py for results visualization.

 7. requirements.txt contains necessary libraries/packages with corresponding versions.

 ## Installation
 
 1. Install dependencies
    
    **pip3 install -r requirements.txt**

 2. Run training part and visualization of results
    
     **python train.py & python test.py**
   
   



# EDA
**train_ship_segmentations_v2.csv** consists of id and encoded pixels, some of them have no encoded pixels, it's mean that there is no ships on the picture. 
![1](https://github.com/yuliadziuba/DS/assets/151251662/bba628e4-8445-4342-89dc-b960788adec1)




After defining the data into those that doesn't contain ships and those on which they are present, we see that the data is very unbalanced.

![2](https://github.com/yuliadziuba/DS/assets/151251662/e71ce21a-0bc3-4fbd-a602-6547f3fcd2fe)


To deal with, we will extract 2000 or less samples per each class (0-15 ships).


![3](https://github.com/yuliadziuba/DS/assets/151251662/02201265-0638-4e28-88f2-1f6e0560cfe7)



After segmentation we can see image's masks and input images. Masks layers are overlain with transparency. These images show that not all inputs are straightforward to classify. There are clouds, shallow water, buildings, wave reflections, and other obstacles that share some characteristics of shipping vessels. The model will need to be robust enough to differentiate between ship and non-ship pixels in these difficult cases. Ships are also a range of sizes, and can sometimes be directly adjacent to one another.

![20](https://github.com/yuliadziuba/DS/assets/151251662/a4f558cd-e445-44e5-a1f5-b79045014463)

More information about data preparation we can see in **run.ipynb**

# Architecture



In this task I use U-Net model - a Convolutional Neural Network developed primarily for segmentation. It is suitable for this problem too cause we need model that takes image as input and also outputs an image.
Model here is fairly simple in order to execute it easily, but also we can create more layers and play with parameters, only looking carefully on the input and output dimensions.
The general architecture of this model is shown below. A series of convolution layers downsample (or encode) the input to the bottleneck, and then layers are upsampled (or decoded) to the output segmentation map.



![Screenshot 2023-11-23 032215](https://github.com/yuliadziuba/DS/assets/151251662/b3ee5b13-33bd-41e5-9a4a-ff2bc45a0e22)


Below we can see which metrics for model estimate I use. Of course, an important part of model building is specifying the loss function. The loss function is used to quantify the error between predicted and actual outputs at each step. The loss function plays a large role in how the model behaves and converges. In this case, I used a combination of the dice and binary cross entropy (the dice_p_bce function).

![Screenshot 2023-11-23 033250](https://github.com/yuliadziuba/DS/assets/151251662/cc071398-37cb-4869-8cd0-cc2b4a4348dc)

The BATCH_SIZE parameter was set to 48 images. The number of steps was 9 in each 99 epochs. The convergence of the model at each epoch can be summarized with the following graphs. The red lines represent the scores on the validation dataset and the blue lines represent the scores on the training dataset. 
As we can see the gap between the training and validation curves is not big, so it can be considered that we managed to avoid overfitting or underfitting the data. But TPR shows that it isn't a high persentage of actual positives which are correctly identified. And also dice score is quite small, which means that segmentation performance was a bit poor.



![6](https://github.com/yuliadziuba/DS/assets/151251662/13bf4380-f18b-43c5-9a80-4f7a9b589bfd)

# Prediction
It would be good to experiment with different loss functions and with a different test/validation ratio, also with image augmentation strategies, because clouds and wave sometimes were identified as ships and the outline of the ship is not entirely clear.
![8](https://github.com/yuliadziuba/DS/assets/151251662/726c60ca-7709-4dad-9f19-cba1e4b32129)

![9](https://github.com/yuliadziuba/DS/assets/151251662/0ffcfd98-0fe1-42de-8500-5ff6251cceed)


![7](https://github.com/yuliadziuba/DS/assets/151251662/b3193fb4-7e52-42c1-82eb-502b165eb141)




