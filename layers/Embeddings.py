from .Base import Layer,Variable
from ..utils.Initializers import get_initializer
import numpy as np



class Embedding(Layer):
    def __init__(self,input_dim,output_dim,embeddings_initializer='uniform',mask_zero=False,input_length=None,**kwargs):
        super(Embedding,self).__init__(**kwargs)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.initializer=get_initializer(embeddings_initializer)
        self.mask_zero=mask_zero
        self.input_length=input_length


    def __call__(self, prev_layer):
        assert self.output_dim is not None
        super(Embedding,self).__call__(prev_layer)
        self._initial_params()
        self.output_shape=self.compute_output_shape()


    def connect(self,inbound_layer):
        assert self.output_dim is not None
        if inbound_layer is None:
            assert self.input_length is not None
            self.input_shape=(None,self.input_length)
        self._initial_params()
        self.output_shape = self.compute_output_shape()
        super(Embedding,self).connect(inbound_layer)





    def _initial_params(self):
        W=Variable(self.initializer((self.input_dim,self.output_dim)),name='embedding_w')
        self.variables.append(W)
        for var in self.variables:
            if var.require_grads:
                var.grads = np.zeros_like(var.output_tensor)



    def compute_output_shape(self):
        return self.input_shape+(self.output_dim,)




    def forward(self,is_training):
        assert self.input_tensor.ndim==2
        W,=self.variables
        #to one-hot

        self.output_tensor=W.output_tensor[self.input_tensor]
        if is_training:
            if not W.require_grads:
                del self.input_tensor
        super().forward(is_training)


    def backward(self):
        W,=self.variables
        flatten_idxs=self.input_tensor.flatten()
        unique_idxs=np.unique(flatten_idxs)
        flatten_grads=self.grads.reshape(-1,self.output_shape[-1])
        if W.require_grads:
            for idx in unique_idxs:
                W.grads[idx]+=np.sum(flatten_grads[flatten_idxs==idx],axis=0)



