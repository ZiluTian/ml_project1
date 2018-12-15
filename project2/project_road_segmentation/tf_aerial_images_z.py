"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zurich
"""

import gzip
import sys
import urllib
import code

# import tensorflow.python.platform
from tf_global_vars import *
from tf_img_helpers import *
from tf_utils import *


tf.app.flags.DEFINE_string('train_dir', '/tmp/segment_aerial_images',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'
    test_data_filename = ['test_set_images/test_'+str(i)+'/test_'+str(i)+'.png' for i in range(1,TESTING_SIZE+1)]

    # Extract train data
    train_data, train_labels, val_data, val_labels = extract_data_labels(train_data_filename, train_labels_filename, TRAINING_SIZE, TRAIN_PER, BORDER)
    num_epochs = NUM_EPOCHS
    numpy.random.seed(NP_SEED)
    tf.set_random_seed(SEED)

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print ('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print (len(new_indices))
    print (train_data.shape)
    train_data = train_data[new_indices,:,:,:]
    train_labels = train_labels[new_indices]

    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))


    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_TOTAL_SIZE, IMG_TOTAL_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    train_all_data_node = tf.constant(train_data)
    eval_data_node = tf.constant(val_data)
    eval_labels_node = tf.constant(val_labels)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.initialize_all_variables().run()}

    conv1_weights = tf.Variable(
        tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    
    conv2_weights = tf.Variable(
        tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    conv3_weights = tf.Variable(
        tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, 64, 128],
                            stddev=0.1,
                            seed=SEED))
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[128]))
    
    conv4_weights = tf.Variable(
    tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, 128, 256],
                        stddev=0.1,
                        seed=SEED))
    conv4_biases = tf.Variable(tf.constant(0.1, shape=[256]))
        
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(256*IMG_TOTAL_SIZE**2 / (2**LAYER_NUMBER)**2), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

#     print("fc1 weights shape", str(fc1_weights.get_shape()))
#     print("fc1 biases shape", str(fc1_biases.get_shape()))
#     print("fc2 weights shape", str(fc2_weights.get_shape()))
#     print("fc2 biases shape", str(fc2_biases.get_shape()))

    # Get prediction for given input image
    def get_prediction(img):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, BORDER))
        data = normalize_img(data)
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

        return img_prediction

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)

        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay_test(filename):

        img = mpimg.imread(filename)

        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg

    def get_prediction_test(filename):

        img = mpimg.imread(filename)

        cimg = img_float_to_uint8(get_prediction(img))

        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg
    
    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        
#         conv_params, output_layer = conv_layers_param(CONV_ARCH, OUTPUT_CHANNELS, NUM_CHANNELS, seed=SEED)
#         conv_end = conv_layers_init(CONV_ARCH, conv_params, data)
        #zt [2,2,1] architecture 
        conv1 = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')  
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        
        conv3 = tf.nn.conv2d(pool, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
        pool2 = tf.nn.max_pool(relu4,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        
        # Reshape the feature map cuboid into a 2D matrix to feed it to the fully connected layers.
        conv_end_shape = pool2.get_shape().as_list()

        reshape = tf.reshape(
            pool2,
            [-1, conv_end_shape[1] * conv_end_shape[2] * conv_end_shape[3]])
        
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.

        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        #if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases

        return out

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True) # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=train_labels_node))
    tf.summary.scalar('loss', loss)

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
#     learning_rate = tf.train.exponential_decay(
#         0.01,                # Base learning rate.
#         batch * BATCH_SIZE,  # Current index into the dataset.
#         train_size,          # Decay step.
#         0.95,                # Decay rate.
#         staircase=True)
    learning_rate = tf.constant(0.001) 
    tf.summary.scalar('learning_rate', learning_rate)

    # Use simple momentum for the optimization.
#     optimizer = tf.train.MomentumOptimizer(learning_rate,
#                                            0.0).minimize(loss,
#                                                          global_step=batch)
    adam_opt = tf.train.AdamOptimizer(learning_rate, beta1=.9, beta2=.999, epsilon=1e-08, use_locking=False, name='Adam')
    optimizer = adam_opt.minimize(loss, global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    init = tf.global_variables_initializer()
    
    with tf.Session() as s:
            
        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            s.run(init)

            # Build the summary operation based on the TF collection of Summaries.
#             summary_op = tf.summary.merge_all()
#             summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
#                                                     graph_def=s.graph_def)
            print ('Initialized!')
            # Loop through training steps.
            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

            training_indices = range(train_size)

            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)
                steps_per_epoch = int(train_size/BATCH_SIZE)

                for step in range (steps_per_epoch):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step == 0:
                        
#                         summary_str, _, l, lr, predictions = s.run(
#                             [summary_op, optimizer, loss, learning_rate, train_prediction],
#                             feed_dict=feed_dict)
#                         summary_writer.add_summary(summary_str, step)
#                         summary_writer.flush()

                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
    
                        # print_predictions(predictions, batch_labels)

                        print ('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk.
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)

        print ("Running prediction on validation set")
        output = tf.nn.softmax(model(eval_data_node))
        output_prediction = s.run(output)
        TP, FP, FN = f1_score(output_prediction,  val_labels)
        precision = TP / (FP + TP)
        recall = TP / (FN + TP)
        f1 = 2 * (precision * recall) / (precision + recall)
        print('F1 score for validation set: %.3f%%' % f1)

        print ("Running prediction on testing set")
        prediction_testing_dir = "predictions_testing/"
        if not os.path.isdir(prediction_testing_dir):
            os.mkdir(prediction_testing_dir)
        for i in range(1, TESTING_SIZE+1):
            pimg = get_prediction_test(test_data_filename[i - 1])
            Image.fromarray(pimg).save(prediction_testing_dir + "prediction_" + str(i) + ".png")
            oimg = get_prediction_with_overlay_test(test_data_filename[i - 1])
            oimg.save(prediction_testing_dir + "overlay_" + str(i) + ".png")
            print("Generated image prediction_" + str(i) + ".png")

if __name__ == '__main__':
    tf.app.run()
