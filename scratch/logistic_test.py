from classification import *
from helpers import standardize

# simple test case
data_path = 'binary.csv'
data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
yd = data[:, 0].astype(np.int)
xd = data[:, 1:]

#make a linear feature matrix
feat, f_mean, f_std = standardize(xd)

#learn the model
w0 = np.zeros(4)
model = logistic_regression(yd, feat, w0, 0.01, 50)

#simple validation
test_path = 'binary_test.csv'
test = np.genfromtxt(test_path, delimiter=",", skip_header=1)
yt = test[:, 0].astype(np.int)
xt = test[:, 1:]

feat_test, f_mean, f_std = standardize(xt)
correct_predictions = 0

for i in range(0, len(yd)):
    x = feat[i, :]
    y_p = logistic_sigmoid(np.dot(x, model))
    if np.rint(y_p) == yd[i]:
        correct_predictions += 1

print('\nPercentage of correct predictions on the training set: {} % \n'.format(100 * correct_predictions / len(yd)))

correct_predictions = 0
for i in range(0, len(yt)):
    x = feat_test[i, :]
    y_p = logistic_sigmoid(np.dot(x, model))
    if np.rint(y_p) == yt[i]:
        correct_predictions += 1

print('Percentage of correct predictions on the validation set: {} % \n'.format(100 * correct_predictions / len(yt)))