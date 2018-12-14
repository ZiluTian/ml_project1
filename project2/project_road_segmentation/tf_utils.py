# General helper functions 
from tf_global_vars import * 

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
    numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])


def f1_score(predictions, labels):
    pred_pos = numpy.argmax(predictions, 1) == 1
    pred_neg = numpy.argmax(predictions, 1) == 0

    n_pos = numpy.argmax(labels, 1) == 1
    n_neg = numpy.argmax(labels, 1) == 0

    TP = numpy.sum(numpy.logical_and(pred_pos, n_pos))
    FP = numpy.sum(numpy.logical_and(pred_pos, n_neg))
    FN = numpy.sum(numpy.logical_and(pred_neg, n_pos))

    return (TP, FP, FN)

def conv_layers_param(conv_arch, conv_depth, channels, seed=None): 
    conv_params = [None] * len(conv_arch)

    input_channel = channels
    
    for i, n_conv in enumerate(conv_arch): 
        conv_params[i] = [None] * n_conv 
        output_channel = conv_depth[i]
        for layer in range(n_conv): 
#             conv_weights = tf.assign(conv_weights, tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, input_channel, output_channel], stddev=0.1, seed=seed), validate_shape=False)
#             conv_biases = tf.assign(conv_biases, tf.zeros([output_channel]), validate_shape=False)
            conv_weights = tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, input_channel, output_channel], stddev=0.1, seed=seed))
            conv_biases = tf.Variable(tf.zeros([output_channel]))
            conv_params[i][layer] = (conv_weights, conv_biases)
#             conv_params[i][layer] = (tf.assign(conv_weights, tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, input_channel, output_channel], stddev=0.1, seed=seed), validate_shape=False), tf.assign(conv_biases, tf.zeros([output_channel]), validate_shape=False))

            input_channel = output_channel 

    return conv_params, output_channel 

# Convolution layers bounded with relus 
def conv_layers_init(conv_arch, conv_params, prev_layer): 
    for i, n_conv in enumerate(conv_arch): 
        for layer in range(n_conv): 
#             conv_weights = tf.assign(conv_biases, conv_params[i][layer][0])
#             conv_biases = conv_params[i][layer][1]
            conv = tf.nn.conv2d(prev_layer, conv_params[i][layer][0], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv_params[i][layer][1]))
            prev_layer = relu 

        prev_layer = tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')

    return prev_layer 


    
# Write predictions from neural network to a file
# def write_predictions_to_file(predictions, labels, filename):
#     max_labels = numpy.argmax(labels, 1)
#     max_predictions = numpy.argmax(predictions, 1)
#     file = open(filename, "w")
#     n = predictions.shape[0]
#     for i in range(0, n):
#         file.write(max_labels(i) + ' ' + max_predictions(i))
#     file.close()

# Print predictions from neural network
# def print_predictions(predictions, labels):
#     max_labels = numpy.argmax(labels, 1)
#     max_predictions = numpy.argmax(predictions, 1)
#     print (str(max_labels) + ' ' + str(max_predictions))
