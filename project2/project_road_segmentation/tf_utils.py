# General helper functions 
import numpy 

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
