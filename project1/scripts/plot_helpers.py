import matplotlib.pyplot as plt

def cross_validation_plot(lambdas, rmse_tr, rmse_te): 
    plt.semilogx(lambdas, rmse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambdas, rmse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()
    return 

def pairwise_correlation_plot(coef_vec): 
    plt.scatter(range(len(coef_vec)), coef_vec)
    plt.xlabel("Feature number")
    plt.ylabel("Absolute value of correlation coefficient")
    plt.title("Correlation coefficient of columns against prediction")
    plt.grid(True)
    plt.show()
    return 

def feature_correlation_plot(corr_matrix): 
    plt.imshow(corr_matrix, interpolation='nearest')
    plt.title("Correlation coef of features")
    plt.colorbar()
    plt.show()
    return 