import numpy as np
from ..layers.Base import Variable


def to_categorical(inputs):
    if inputs.ndim > 2:
        raise ValueError('only accept 1-d or 2-d inputs')
    # convert Y to one-hot encode
    # for example,merge (batch,1) to (batch,)
    if inputs.ndim == 2:
        if inputs.shape[-1] == 1:
            inputs = np.sum(inputs, axis=-1)
        else:
            raise ValueError('can not convert %s to one-hot vector' % (inputs.__class__))

    n_class = np.max(inputs).tolist() + 1
    encoded_Y = np.eye(n_class)[inputs].reshape(-1, n_class)
    return encoded_Y




def concatenate(*variables,axis,output=None,name=None):
    new_shape=np.asarray(variables[0].output_shape)
    new_value=variables[0].output_tensor
    for i in range(1,len(variables)):
        shape=np.asarray(variables[i].output_shape)
        new_shape[axis]+=shape[axis]
        new_value=np.concatenate((new_value,variables[i].output_tensor),axis=axis)
    if output is None:
        newVar=Variable(initial_value=new_value,shape=new_shape.tolist(),name=name)
    else:
        output.output_tensor=new_value
        output.output_shape=new_shape
        output.name=name
        newVar=output

    return newVar



def pad_sequences(sequences,maxlen=None,dtype='int32',padding='pre',truncating='pre',value=0.):
    lengths=[]
    for x in sequences:
        lengths.append(len(x))
    num_samples=len(sequences)
    if maxlen is None:
        maxlen=max(lengths)
    # lengths=list(map(lambda x,y:y if x>y else x,lengths,[maxlen]*len(lengths)))

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)

    for idx,s in enumerate(sequences):
        if not len(s):
            continue

        if truncating=='pre':
            trunc=s[-maxlen:]
        elif truncating=='post':
            trunc=s[:maxlen]
        else:raise ValueError('unknown truncating type!')

        trunc=np.asarray(trunc,dtype=dtype)

        if padding=='pre':
            x[idx,-len(trunc):]=trunc
        elif padding=='post':
            x[idx,:len(trunc)]=trunc
        else:raise ValueError('unknown padding type!')
    return x




