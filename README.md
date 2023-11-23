# Summary
This repo details my solution to [Kaggle's Airbus Ship Detection](https://www.kaggle.com/competitions/airbus-ship-detection) challenge. The aim of the competition is to analyze satellite images, detect the ships 
and produce segmentation masks of the ships. We have original images (around 100 k of them) and corresponding masks for each occuring ship in .csv format. Also this repo includes a pipeline for training UNet model for 
the problem of ships detection. I will use notebooks **run.ipynb** as main file in EDA below, all images from output **run.ipynb**. 

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
Those data are in the form of ImageId -> RLE Encoded Vector. The output of the images in dataset are encoded using Run Length Encoding (RLE), which based on [script]
(https://www.kaggle.com/code/paulorzp/run-length-encode-and-decode/script). Also the folders with images you can find in **train_v2** and **test_v2**

## Files description:

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
    
     or
    
     **python run.ipynb**
   
   



# EDA

