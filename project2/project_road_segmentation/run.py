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

    conv1_weights = tf.Variable(
    tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                        stddev=0.1,
                        seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))

    conv12_weights = tf.Variable(
        tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, 32, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv12_biases = tf.Variable(tf.zeros([32]))

    conv2_weights = tf.Variable(
        tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    conv22_weights = tf.Variable(
        tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, 64, 64],
                            stddev=0.1,
                            seed=SEED))
    conv22_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    
    def model(data):
        """The Model definition."""
#zt Elegant solution, only if it works :/
#         layer1 = create_layer(data, NUM_CHANNELS, 32) # depth 32 
#         layer2 = create_layer(layer1, 32, 64)
        conv1 = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        conv12 = tf.nn.conv2d(relu1, conv12_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu12 = tf.nn.relu(tf.nn.bias_add(conv12, conv12_biases))

        pool = tf.nn.max_pool(relu12,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        conv2 = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        conv22 = tf.nn.conv2d(relu2, conv22_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu22 = tf.nn.relu(tf.nn.bias_add(conv22, conv22_biases))

        layer2 = tf.nn.max_pool(relu22,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        
        conv_end_shape = layer2.get_shape().as_list()
        reshape = tf.reshape(
            layer2,
            [conv_end_shape[0], conv_end_shape[1] * conv_end_shape[2] * conv_end_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, fc1['weights']) + fc1['biases'])
        out = tf.matmul(hidden, fc2['weights']) + fc2['biases']
        return out

    # Get the loss of the model over the train data 
    logits = model(train_data_node)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=train_labels_node))
    tf.summary.scalar('loss', loss)

    regularizers = (tf.nn.l2_loss(fc1['weights']) + tf.nn.l2_loss(fc1['biases']) + tf.nn.l2_loss(fc2['weights']) + tf.nn.l2_loss(fc2['biases']))
    loss += 5e-4 * regularizers
    batch = tf.Variable(0)
    learning_rate = tf.constant(0.001)
    adam_opt = tf.train.AdamOptimizer(learning_rate, beta1=.9, beta2=.999, epsilon=1e-08, use_locking=False, name='Adam')
    optimizer = adam_opt.minimize(loss, global_step=batch)

    # Get the loss of the model over the evaluation data 
    train_prediction = tf.nn.softmax(logits)
    predictions = tf.nn.softmax(model(eval_data_node))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(eval_labels_node, 1)), tf.float32))
    train_all_prediction = tf.nn.softmax(model(train_all_data_node))
    saver = tf.train.Saver()

    with tf.Session() as s:

        if RESTORE_MODEL:
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            tf.global_variables_initializer().run()

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
            f1 = f1_score(output_prediction,  val_labels)
            print('F1 score for validation set: %.3f' % f1)
            params['f1'] = f1

        if PREDICT_IMAGES:
            predict_images(model, s)
            
        if LOG_PARAM:
            write_log(params) 

if __name__ == '__main__':
    tf.app.run()
