# common libaries
import tensorflow as tf
import numpy


NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
TESTING_SIZE = 50
TRAIN_PER = .8  # Size of the training set.
SEED = 66478  # Set to None for random seed.
NP_SEED= 78456
BATCH_SIZE = 64 # 64
NUM_EPOCHS = 10
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
FILTER_SIZE = 3
IMG_PATCH_SIZE = 16
BORDER = 0
IMG_TOTAL_SIZE = IMG_PATCH_SIZE + 2*BORDER
LAYER_NUMBER = 2
PREDICT_IMAGES = True
LOG_PARAM = True
PREDICT_F1 = True
# CONV_ARCH = [2, 2]
