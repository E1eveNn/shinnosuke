import numpy as np


class regularizers():
    def __call__(self, x):
        return 0.



class L1L2(regularizers):
    def __init__(self,l1=0.,l2=0.):
        self.l1=l1
        self.l2=l2


    def __call__(self, x):
        regularization=0.
        if self.l1:
            regularization+=np.sum(self.l1*np.abs(x))/x.shape[0]
        if self.l2:
            regularization+=np.sum(self.l1*np.square(x))/(2*x.shape[0])
        return regularization


def l1(lambd):
    return L1L2(l1=lambd)


def l2(lambd):
    return L1L2(l2=lambd)



def get_regularizer(regularizer):
    if regularizer is None:
        return None
    elif callable(regularizer):
        return regularizer
    else:
        raise ValueError('unknown regularizer type!')

