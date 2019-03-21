import numpy as np

def augment_features_window(X, N_neig):
    N_row = X.shape[0]
    N_feat = X.shape[1]
    X = np.vstack((np.zeros((N_neig, N_feat)),np.zeros((N_neig, N_feat)), X, np.zeros((N_neig, N_feat)),np.zeros((N_neig, N_feat))))
    X_aug = np.zeros((N_row, N_feat*(4*N_neig+1)))
    for r in np.arange(N_row) + N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
            if c != 0:
                #print(len((X[r] + X[r+c])/2))
                this_row = np.hstack((this_row, (X[r] + X[r+c])/2))
        #print(len(this_row))
        X_aug[r-N_neig] = this_row

    return X_aug

def augment_features_gradient(X, depth):
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad

def augment_features(X, well, depth, N_neig=1):
    X_aug = np.zeros((X.shape[0], X.shape[1]*(4*N_neig+1)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        #print(X_aug_win)
        #X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        #print(X_aug_grad)
        X_aug[w_idx, :] = X_aug_win
        #X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
        
    return X_aug

