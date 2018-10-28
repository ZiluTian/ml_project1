# import all the functions from the helper module 
from proj1_helpers import *
from implementations import *
# from proj1_plot_helpers import * 
from proj1_feature_selection import *


# A helper function that loads the data, fills in the missing values with average, 
# and normalize data
print("Prepare to load data")
y, inputs, ids = load_clean_csv("train.csv", sub_sample=False, missing_val="avg", normalized="True")
print("Data preprocessing finished")
# Apply stepwise regression for feature selection 
feature_list, scores = stepwise_regression(inputs,y)
print("Feature selection finished")
# Select only the 10 best features and sort it by name
feature_list=feature_list[:10]
feature_list.sort()

# Focus on selected features 
x = inputs[:, feature_list]
degree = 3
k_fold = 4

# Build polynomial model of the given degree. Different combinations are also considered
tx = build_poly_plus(x, degree)
print("Building polynomial model finished")
# Find an optimal lambda (with least rmse) from a specified space using grid search 
lambdas = np.logspace(-5, 0, 30)
opt_lambda, rmse_tr, rmse_te = find_desired_var(lambdas, y, tx, k_fold, ridge_regression)
print("Finding lambda for ridge regression finished")
# find_weight applies cross validation by splitting data k_fold and  
# the final weight matrix is the average over matrices that result in 
# least rmse for each run 
w, mse = find_weight(y, tx, k_fold, ridge_regression, opt_lambda)
# compute_score(y_test, y_pred)
print("Finding weight matrix finished")
# Load, clean, and normalize testing data 
y_test, inputs_test, ids_test = load_clean_csv('test.csv', False, "avg", True)
# Build model using the same feature list
tx_test = build_poly_plus(inputs_test[:,feature_list], degree)
# Apply the trained label over the test data 
y_pred = predict_labels(w, tx_test)
print("Prediction finished")
# Create submission 
create_csv_submission(ids_test, y_pred, "prediction.csv")
print("Creating submission file finished")

# Optional (Supplementary for data used in report)
# ================
# Alternative code for feature selection
# coef_vec = pairwise_correlation(y, inputs)
# pairwise_correlation_plot(coef_vec)

# feature_threshold = 0.1 
# feature_list = feature_select(coef_vec, feature_threshold)
# corr_matrix = correlation_matrix(inputs, feature_list)

# # remove from feature list features with correlation coef higher than threhold
# duplicate_threshold = 0.8

# feature_list = feature_extract(feature_list, corr_matrix, duplicate_threshold)
# corr_matrix = correlation_matrix(inputs, feature_list)

# ================
# Code for comparing different regression methods
# lambdas = np.logspace(-5, 0, 30)
# gammas = np.logspace(-5, 0, 20)

# init_w = np.zeros(tx.shape[1])
# max_it = 50

# opt_lambda, rmse_tr, rmse_te = find_desired_var(lambdas, y, tx, k_fold, ridge_regression)
# print("ridge regression done. opt_lambda is", opt_lambda)
# opt_gamma_1, rmse_tr1, rmse_te1 = find_desired_var(gammas, y, tx, k_fold, logistic_regression, init_w, max_it)
# print("logistic regression done. opt_gamma is", opt_gamma_1)
# opt_gamma_2, rmse_tr2, rmse_te2 = find_desired_var(gammas, y, tx, k_fold, least_squares_GD, init_w, max_it)
# print("least squares GD done. opt_gamma is", opt_gamma_2)
# opt_gamma_3, rmse_tr3, rmse_te3 = find_desired_var(gammas, y, tx, k_fold, least_squares_SGD, init_w, max_it)
# print("least squares SGD done. opt_gamma is", opt_gamma_3)

