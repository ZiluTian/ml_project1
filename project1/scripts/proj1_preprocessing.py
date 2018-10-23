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

