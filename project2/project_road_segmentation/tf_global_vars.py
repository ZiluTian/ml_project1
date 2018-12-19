# common libaries
import tensorflow as tf
import numpy


NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2 # Binary classification 
TRAINING_SIZE = 100 # Shouldn't exceed the image number in folder /training 
TESTING_SIZE = 50 # Shouldn't exceed the image number in folder /testing 
TRAIN_PER = .8  # Percentage of the training set used for training vs evaluation.
SEED = 66478  # Set to None for random seed.
NP_SEED= 78456
BATCH_SIZE = 64 # 16
NUM_EPOCHS = 5 
RESTORE_MODEL = True # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
FILTER_SIZE = 3 # Experimentally tested as a good fit 
IMG_PATCH_SIZE = 16 # Should be a multiple of 8 
BORDER = 8 # Should be a multiple of 8. Neighbor pixels outside the patch used for classification
IMG_TOTAL_SIZE = IMG_PATCH_SIZE + 2*BORDER
LAYER_NUMBER = 2 # The current layer number used for neural net training. 
PREDICT_IMAGES = True # If True, generate predicted/overlay images 
LOG_PARAM = True # If True, log the program to log_file.txt in the current folder
PREDICT_F1 = True # If True, print out predicted F1 score 
