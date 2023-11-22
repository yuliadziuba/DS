from config import *
import os
import numpy as np 
import keras
import tensorflow as tf
import pandas as pd 
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from skimage.segmentation import mark_boundaries
from skimage.util import montage as montage2d
from utils import utils, loss, generator
from skimage.morphology import binary_opening, disk, label
from tqdm.notebook import tqdm
from utils import utils, loss, generator, predict

#Model Parameters
ship_dir = '..\DS\data'
test_image_dir = os.path.join(ship_dir, 'test_v2')

test_paths = np.array(os.listdir(test_image_dir))
print(len(test_paths), 'test images found')

#Create submission
out_pred_rows = []
for c_img_name in tqdm(test_paths[:30000]): ## only a subset as it takes too long to run
    out_pred_rows += predict.pred_encode(c_img_name, min_max_threshold=1.0)

sub = pd.DataFrame(out_pred_rows)
sub.columns = ['ImageId', 'EncodedPixels']
sub = sub[sub.EncodedPixels.notnull()]
sub.head()

#Let's show predicted images
TOP_PREDICTIONS=10
fig, m_axs = plt.subplots(TOP_PREDICTIONS, 2, figsize = (9, TOP_PREDICTIONS*5))
[c_ax.axis('off') for c_ax in m_axs.flatten()]

for (ax1, ax2), c_img_name in zip(m_axs, sub.ImageId.unique()[:TOP_PREDICTIONS]):
    c_img = imread(os.path.join(test_image_dir, c_img_name))
    c_img = np.expand_dims(c_img, 0)/255.0
    ax1.imshow(c_img[0])
    ax1.set_title('Image: ' + c_img_name)
    ax2.imshow(utils.masks_as_color(sub.query('ImageId=="{}"'.format(c_img_name))['EncodedPixels']))
    ax2.set_title('Prediction')

sub1 = pd.read_csv( os.path.join(ship_dir, 'sample_submission_v2.csv'))
sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])
sub1['EncodedPixels'] = None
print(len(sub1), len(sub))

sub = pd.concat([sub, sub1])
print(len(sub))
sub.to_csv('submission.csv', index=False)
sub.head()