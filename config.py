#Some hyperparameter optimizations 
BATCH_SIZE = 48
EDGE_CROP = 16
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
#Downsampling inside the network
NET_SCALING = (1, 1)
#Downsampling in preprocessing
IMG_SCALING = (3, 3)
#Number of validation images to use
VALID_IMG_COUNT = 900
#Maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 9
MAX_TRAIN_EPOCHS = 99
AUGMENT_BRIGHTNESS = False