# common libaries 
import tensorflow as tf
import numpy 


NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 10
TESTING_SIZE = 50
TRAIN_PER = .7  # Size of the training set.
SEED = 66478  # Set to None for random seed.
NP_SEED= 78456
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 1
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 0
FILTER_SIZE = 3
# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16 
BORDER = 8
IMG_TOTAL_SIZE = IMG_PATCH_SIZE + 2 * BORDER

CONV_ARCH = [1, 1]
OUTPUT_CHANNELS = [32, 64]

