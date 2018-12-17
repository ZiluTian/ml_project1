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

def create_layer(data, input_channels, depth): 
    conv1_weights = tf.Variable(
            tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, input_channels, depth],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([depth]))

    conv2_weights = tf.Variable(
        tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, depth, depth],  
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.zeros([depth]))

    conv1 = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    pool = tf.nn.max_pool(relu2,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    
    return pool 
    
    
def write_log(params): 
        log_file_name = "log_file.txt"
        log_file = open(log_file_name, 'a')
        log_file.write('\n\n')
        for i, j in params.items():
            log_file.write(i + ":" + str(j) + "\n")
        log_file.close()
            
                