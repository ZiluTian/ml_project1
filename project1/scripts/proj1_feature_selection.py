# Functions used/tested for feature selection 
import numpy as np
from implementations import *

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
    return [f for f in feature_list if f not in dup_features]


def stepwise_regression(inputs, y):
    """Perform forward selection, where 1st order models are evaluated using least squares resulting MSE score. The function returns a vector of features ordered by their contribution to reducing the MSE score."""
    queue = list(range(inputs.shape[1]))
    selected = []
    feats = []
    current_score, best_score = 10.0e8, 10.0e8
    
    while any(queue) and current_score == best_score:
        scores_candidates = []
        
        # Test all feature-candidates
        for candidate in queue:
            feats = selected + [candidate]
            poly_basis = build_poly(inputs[:, feats], 1)
            _, score = least_squares(y, poly_basis)
            scores_candidates.append((score, candidate))
        
        # Selects the best feature-candidate
        scores_candidates.sort(reverse = True)
        best_score, best_candidate = scores_candidates.pop()
        
        # Keeps the feature in the model in case the score has improved
        if current_score > best_score:
            queue.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_score
    feats = selected
    return feats
