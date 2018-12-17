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
params = {}

def main(argv=None):  # pylint: disable=unused-argument

    numpy.random.seed(NP_SEED)
    tf.set_random_seed(SEED)
    
    test_data_filename = ['test_set_images/test_'+str(i)+'/test_'+str(i)+'.png' for i in range(1,TESTING_SIZE+1)]
    
    train_data_origin, train_labels_origin, val_data, val_labels = extract_train_data(TRAINING_SIZE, TRAIN_PER, BORDER)
    
    train_data, train_labels = balance_data(train_data_origin, train_labels_origin)
    train_size = train_labels.shape[0]
    
    num_epochs = NUM_EPOCHS

    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_TOTAL_SIZE, IMG_TOTAL_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    train_all_data_node = tf.constant(train_data)
    eval_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_TOTAL_SIZE, IMG_TOTAL_SIZE, NUM_CHANNELS))
    eval_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))

    fc1 = {'weights': tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(64*IMG_TOTAL_SIZE**2 / (2**LAYER_NUMBER)**2), 512],
                            stddev=0.1,
                            seed=SEED)), 
           'biases': tf.Variable(tf.constant(0.1, shape=[512]))}
    
    fc2 = {'weights': tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED)), 
           'biases': tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))}   

    def get_prediction(img):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, BORDER))
        data = normalize_img(data)
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

        return img_prediction

    def get_prediction_with_groundtruth(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)

        return cimg

    def get_prediction_with_overlay_test(filename):

        img = mpimg.imread(filename)

        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg

    def get_prediction_test(filename):

        img = mpimg.imread(filename)

        cimg = img_float_to_uint8(get_prediction(img))

        return cimg

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
        
        layer1 = create_layer(data, NUM_CHANNELS, 32) # depth 32 
        layer2 = create_layer(layer1, 32, 64) 

        conv_end_shape = layer2.get_shape().as_list()

        reshape = tf.reshape(
            layer2,
            [conv_end_shape[0], conv_end_shape[1] * conv_end_shape[2] * conv_end_shape[3]])

        hidden = tf.nn.relu(tf.matmul(reshape, fc1['weights']) + fc1['biases'])
        #if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2['weights']) + fc2['biases']

        return out


    logits = model(train_data_node, True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=train_labels_node))
    tf.summary.scalar('loss', loss)

    regularizers = (tf.nn.l2_loss(fc1['weights']) + tf.nn.l2_loss(fc1['biases']) + tf.nn.l2_loss(fc2['weights']) + tf.nn.l2_loss(fc2['biases']))
    loss += 5e-4 * regularizers

    batch = tf.Variable(0)

    learning_rate = tf.constant(0.001)

    adam_opt = tf.train.AdamOptimizer(learning_rate, beta1=.9, beta2=.999, epsilon=1e-08, use_locking=False, name='Adam')
    optimizer = adam_opt.minimize(loss, global_step=batch)

    train_prediction = tf.nn.softmax(logits)
    predictions = tf.nn.softmax(model(eval_data_node))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(eval_labels_node, 1)), tf.float32))

    train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    with tf.Session() as s:

        if RESTORE_MODEL:
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            s.run(init)

            print("\n############################################################################")
            print ('Training CNN')
            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

            training_indices = range(train_size)
            log_acc_test = []
            log_acc_train = []

            for iepoch in range(num_epochs):

                perm_indices = numpy.random.permutation(training_indices)
                steps_per_epoch = int(train_size/BATCH_SIZE)

                for step in range (steps_per_epoch):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    batch_data = train_data[batch_indices]
                    batch_labels = train_labels[batch_indices]

                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step % RECORDING_STEP == 0:

                        _, l = s.run([optimizer, loss], feed_dict=feed_dict)

                        ave_acc_train = epoch_eval(s, accuracy, train_data, train_labels, BATCH_SIZE, eval_data_node, eval_labels_node)
                        log_acc_train.append(ave_acc_train)

                        ave_acc_test = epoch_eval(s, accuracy, val_data, val_labels, BATCH_SIZE, eval_data_node, eval_labels_node)
                        log_acc_test.append(ave_acc_test)

                        print ('Epoch ' + str(iepoch + 1) + ' %.2f ' % (float(step) * BATCH_SIZE / train_size))
                        print ('Minibatch loss: %.3f ' % l)
                        print ('Train accuracy: %.2f%%' % ave_acc_train)
                        print ('Test accuracy: %.2f%%' % ave_acc_test)

                        sys.stdout.flush()
                    else:
                        _, l = s.run([optimizer, loss],feed_dict=feed_dict)

                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)

        params['Average_acc_train'] = log_acc_train
        params['Average_acc_test'] = log_acc_test
        params['Batch_size'] = BATCH_SIZE
        params['N_epochs'] = NUM_EPOCHS
        params['Filter_size'] = FILTER_SIZE
        params['Img_patch_size'] = IMG_PATCH_SIZE
        params['Border'] = BORDER

        if PREDICT_F1:
            print("\n############################################################################")
            print ("Running prediction on validation set")
            output = tf.nn.softmax(model(tf.constant(val_data)))
            output_prediction = s.run(output)
            TP, FP, FN = f1_score(output_prediction,  val_labels)
            precision = TP / (FP + TP)
            recall = TP / (FN + TP)
            f1 = 2 * (precision * recall) / (precision + recall)
            print('F1 score for validation set: %.3f' % f1)
            params ['f1'] = f1

        if PREDICT_IMAGES:
            print("\n############################################################################")
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

        if LOG_PARAM:
            log_file_name = "log_file.txt"
            log_file = open(log_file_name, 'a')
            log_file.write('\n\n')
            for i, j in params.items():
                log_file.write(i + ":" + str(j) + "\n")
            log_file.close()

if __name__ == '__main__':
    tf.app.run()
