import numpy as np

def make_dataset_iterator(X, y, batch_size):
    pass  #TODO


# Subfuction to normalize mixing matrix
def l2normalize(Amat, axis=0):
    # axis: 0=column-normalization, 1=row-normalization
    l2norm = np.sqrt(np.sum(Amat * Amat, axis, keepdims=True))
    Amat = Amat / l2norm
    return Amat
