# General helper functions
from tf_global_vars import *


def f1_score(predictions, labels):
    pred_pos = numpy.argmax(predictions, 1) == 1
    pred_neg = numpy.argmax(predictions, 1) == 0

    n_pos = numpy.argmax(labels, 1) == 1
    n_neg = numpy.argmax(labels, 1) == 0

    TP = numpy.sum(numpy.logical_and(pred_pos, n_pos))
    FP = numpy.sum(numpy.logical_and(pred_pos, n_neg))
    FN = numpy.sum(numpy.logical_and(pred_neg, n_pos))

    return (TP, FP, FN)

def balance_data(train_data, train_labels): 
    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1

    print ('Number of data points per class: Background = ' + str(c0) + ' Road = ' + str(c1))
    
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
    print ('Number of data points per class: Background = ' + str(c0) + ' Road = ' + str(c1))
    return (train_data, train_labels)

def tf_nodes_declare(): 
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_TOTAL_SIZE, IMG_TOTAL_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    eval_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_TOTAL_SIZE, IMG_TOTAL_SIZE, NUM_CHANNELS))
    eval_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    
    return (train_data_node, train_labels_node, eval_data_node, eval_labels_node)

# def tf_var_declare(): 
#     """Declare layer parameters and data nodes"""

#     conv1_weights = tf.Variable(
#         tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
#                             stddev=0.1,
#                             seed=SEED))
#     conv1_biases = tf.Variable(tf.zeros([32]))

#     conv12_weights = tf.Variable(
#         tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, 32, 32],  # 5x5 filter, depth 32.
#                             stddev=0.1,
#                             seed=SEED))
#     conv12_biases = tf.Variable(tf.zeros([32]))

#     conv2_weights = tf.Variable(
#         tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, 32, 64],
#                             stddev=0.1,
#                             seed=SEED))
#     conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

#     conv22_weights = tf.Variable(
#         tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, 64, 64],
#                             stddev=0.1,
#                             seed=SEED))
#     conv22_biases = tf.Variable(tf.constant(0.1, shape=[64]))

#     fc1_weights = tf.Variable(  # fully connected, depth 512.
#         tf.truncated_normal([int(64*IMG_TOTAL_SIZE**2 / (2**LAYER_NUMBER)**2), 512],
#                             stddev=0.1,
#                             seed=SEED))
#     fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

#     fc2_weights = tf.Variable(
#         tf.truncated_normal([512, NUM_LABELS],
#                             stddev=0.1,
#                             seed=SEED))
#     fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
    
#     return (conv1_weights, conv1_biases, conv12_weights, conv12_biases, conv2_weights, conv2_biases, conv22_weights, conv22_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases)
