# Helper functions related to data preprocessing and feature selection 
import numpy as np
from proj1_helpers import load_csv_data  

def get_missing_index(input_data):
    """Retrieve the indices of missing values in input_data. Missing values are denoted by -999"""
    missing_ind = (input_data[:,4]==-999)
    col_num = input_data.shape[1]
    for col in range(col_num): 
        missing_ind = (missing_ind | (input_data[:,col]==-999))
    return missing_ind 

def load_clean_csv(data_path, sub_sample=False, missing_val="ignore", normalized=True): 
    """Load clean csv, specify data_path, sub_sample(True/False), missing_val(ignore, avg, median), normalized(True/False)
    Return yb, input_data, and ids"""
    yb, input_data, ids = load_csv_data(data_path, sub_sample)
    missing_ind = get_missing_index(input_data)
    
    incomplete_features = np.unique(np.where(input_data == -999.0)[1])
    
    if (missing_val=="avg"): 
        mean = np.mean(input_data[~missing_ind], 0)
        for i in incomplete_features:
            np.place(input_data[:,i], input_data[:,i] == -999, mean[i])
    elif (missing_val=="median"): 
        median = np.median(input_data[~missing_ind], 0)
        for i in incomplete_features:
            np.place(input_data[:,i], input_data[:,i] == -999, median[i])
    else:  
        yb = yb[~missing_ind]
        input_data = input_data[~missing_ind]
        ids = ids[~missing_ind]
        
    if normalized: 
        input_m = np.mean(input_data,0)
        input_std = np.std(input_data, 0)
        input_data = (input_data - input_m)/input_std
    
    return yb, input_data, ids

def pairwise_correlation(y, input_data): 
    """Compute pairwise correlation coefficient"""
    coef_vec = []
    for column in range(input_data.shape[1]): 
        coef_vec.append(np.corrcoef(y, input_data[:,column])[0,1])
    return np.abs(coef_vec)

def correlation_matrix(input_data, feature_list): 
    """Compute the correlation matrix"""
    feature_num = len(feature_list)
    corr_matrix = np.array([[0.1 for i in range(feature_num)] for j in range(feature_num)])
    for i in range(feature_num): 
        foo = feature_list[i]
        for j in range(feature_num): 
            bar = feature_list[j]
            corr_matrix[i][j]=np.corrcoef(input_data[:,foo], input_data[:,bar])[0,1]
       
    return np.abs(corr_matrix)

def feature_select(coef_vec, feature_threshold): 
    """Return a list of features with correlation coefficient above feature threshold"""
    return [i for i, x in enumerate(coef_vec) if x>feature_threshold]

def feature_extract(feature_list, corr_matrix, duplicate_threshold):
    """Extract from feature list features that are NOT highly correlated based on duplicate threshold"""
    # Filter out the indices of highly correlated features in corr_matrix 
    repeated_ind = np.where(corr_matrix>duplicate_threshold)
    dup_ind = np.unique(list(np.array(repeated_ind[1])[repeated_ind[0]-repeated_ind[1]!=0]))
    dup_features = [feature_list[i] for i in dup_ind]
    return [feature for feature in feature_list if feature not in dup_features]