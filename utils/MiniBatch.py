import numpy as np




def get_batches(X,y,batch_size,seed,shuffle):
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []
    # 第一步打乱X,Y
    if shuffle:
        permutation = np.random.permutation(m) # 返回一个长度为m的list，里面的值为0到m-1
        shuffled_X = X[permutation]
        shuffled_y = y[permutation]
    else:
        shuffled_X=X
        shuffled_y=y

    # 第二步分割数据
    complete_batch_nums = m // batch_size  # 完整的minibatch个数

    for i in range(complete_batch_nums):
        mini_batch_X = shuffled_X[batch_size * i:batch_size * (i + 1)]
        mini_batch_y = shuffled_y[batch_size * i:batch_size * (i + 1)]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    if m % batch_size != 0:
        mini_batch_X = shuffled_X[ batch_size * complete_batch_nums:]
        mini_batch_y = shuffled_y[ batch_size * complete_batch_nums:]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches