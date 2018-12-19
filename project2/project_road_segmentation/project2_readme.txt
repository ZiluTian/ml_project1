Project 2, Option A Road Segmentation 

Team members: Virginia Bordignon, Zilu Tian, and Tatiana Volkova 

Due: December 20, 2018 

Here is a list of necessary scripts and data for the project which should be present in this folder:
	run.py
	tf_img_helpers.py 
	tf_utils.py 
	tf_global_vars.py  
	mask_to_submission.py 
	test_set_images/ 
	training/ 

Please make sure you have external python libraries Tensorflow, numpy, and matplotlib installed. To get started, open a terminal and enter 
	python run.py 

The default behavior of the script is to generate a new model based on training data rather than restoring an existing model from a checkpoint. This can be changed by setting the corresponding variable RESTORE_MODEL in tf_global_vars.py, which contains the values for all global variables. Other tunable variables include TRAINING_SIZE, TESTING_SIZE, TRAIN_PER, BATCH_SIZE, NUM_EPOCHS, IMG_PATCH_SIZE, BORDER, PREDICT_IMAGES, LOG_PARAM, and PREDICT_F1. These variables are self-explanatory and can be adjusted for different purposes (debug, test, deploy) before running the script. More details can be found in the comment in file tf_global_vars.py

The program first preprocesses the training images defined in folder training/, which include cropping images into lists of patches, converting values to appropriate numerical representation and normalizing them, and extract training labels. Script tf_img_helpers.py contains the preprocessing functions and should be referred to for more explanations. The training images are separated according to TRAIN_PER into validation data and training data. The training data is then balanced based on the number of elements in each label class. 

The next stage is to build a CNN. The model definition and values used for each layer can be found in run.py. The loss of the model is calculated using cross entropy, and is minimized using Adam optimizer at a constant learning rate. The helper functions are defined in tf_utils.py. After the model has been built, we apply it over the evaluation set and get the accuracy result. Please refer to comments in run.py for in-depth explanation. 

After calibrating our model, we can apply it over the testing images in folder test_set_images/. This will generate binary prediction images in folder predictions_testing/ which are used for submission. To generate the submission file based on prediction images, run 
	python mask_to_submission.py 
which will generate file 
	prediction.csv 
in the current directory. 

