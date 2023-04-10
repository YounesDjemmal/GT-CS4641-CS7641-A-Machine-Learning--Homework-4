import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    
    #TODO: implement this
    xmean = np.mean(X[:, 0])
    ymean = np.mean(X[:, 1])
    z1 = X[:, 0] - xmean
    z2 = X[:, 1] - ymean
    f3 = z1 * z2
    f3 = f3[:, np.newaxis]
    X = np.append(X, f3, axis=1)
    return X



    
