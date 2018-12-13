NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
TESTING_SIZE = 50
TRAIN_PER = .6  # Size of the training set.
SEED = 66478  # Set to None for random seed.
NP_SEED = 100
BATCH_SIZE = 128 # 64
NUM_EPOCHS = 10
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 0
FILTER_SIZE = 3
# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
BORDER = 8
LAYER_NUMBERS = 2
IMG_TOTAL_SIZE = IMG_PATCH_SIZE + 2 * BORDER
