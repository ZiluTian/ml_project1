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
    
    # In case we are interested in other metrics 
    precision = TP / (FP + TP)
    recall = TP / (FN + TP)
    
    f1 = 2 * (precision * recall) / (precision + recall)
            
    return f1 

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
   

def split_helper(data_len, bins): 
    return numpy.array_split(range(data_len), bins)

def epoch_eval(s, accuracy, train_data_set, train_data_labels, eval_data_set, eval_data_labels, data_node, label_node):
    set_len = {'train':len(train_data_set), 'eval':len(eval_data_set)}
    batch_nbr = {'train':int(set_len['train'] / BATCH_SIZE) + 1, 'eval':int(set_len['eval'] / BATCH_SIZE) + 1}
    batch_idxs = {'train':split_helper(set_len['train'], batch_nbr['train']), 'eval':split_helper(set_len['eval'], batch_nbr['eval'])}
    acc_train = 0 
    for batch_idx in batch_idxs['train']:
        if len(batch_idx) < BATCH_SIZE:
            batch_idx = range(set_len['train'])[-BATCH_SIZE:]
        feed_dict = {data_node: train_data_set[batch_idx],
                     label_node: train_data_labels[batch_idx]}
        acc_train += s.run(accuracy, feed_dict=feed_dict)
    ave_acc_train = acc_train/batch_nbr['train']*100

    acc_eval = 0 
    for batch_idx in batch_idxs['eval']:
        if len(batch_idx) < BATCH_SIZE:
            batch_idx = range(set_len['eval'])[-BATCH_SIZE:]
        feed_dict = {data_node: eval_data_set[batch_idx],
                     label_node: eval_data_labels[batch_idx]}
        acc_eval += s.run(accuracy, feed_dict=feed_dict)
    ave_acc_eval = acc_eval/batch_nbr['eval']*100

    return (ave_acc_train, ave_acc_eval)


def log_param(params): 
    log_file_name = "log_file.txt"
    log_file = open(log_file_name, 'a')
    log_file.write('\n\n')
    for i, j in params.items():
        log_file.write(i + ":" + str(j) + "\n")
    log_file.close()

def print_train_start(train_size): 
    print("\n############################################################################")
    print ('Training CNN')
    print ('Total number of iterations = ' + str(int(NUM_EPOCHS * train_size / BATCH_SIZE)))

def print_train_epoch(iepoch, loss, ave_acc_train, ave_acc_test): 
    print ('Epoch ' + str(iepoch + 1))
    print ('Minibatch loss: %.3f ' % loss)
    print ('Train accuracy: %.2f%%' % ave_acc_train)
    print ('Test accuracy: %.2f%%' % ave_acc_test)

def predict_f1(s, model, val_data, val_labels): 
    print("\n############################################################################")
    print ("Running prediction on validation set")
    output = tf.nn.softmax(model(tf.constant(val_data)))
    output_prediction = s.run(output)
    f1 = f1_score(output_prediction,  val_labels)
    print('F1 score for validation set: %.3f' % f1)
    return f1 

                