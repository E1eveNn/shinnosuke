from .Base import Layer,Variable
from ..utils.Initializers import get_initializer
from ..utils.Activator import get_activator
from ..utils.Regularizers import get_regularizer
import numpy as np




class Flatten(Layer):
    def __init__(self,out_dim=2):
        if out_dim<1:
            raise ValueError('out_dim must be > 0')
        self.out_dim=out_dim
        super(Flatten,self).__init__()



    def connect(self,prev_layer):
        assert len(prev_layer.output_shape)>=3
        flatten_shape=np.prod(np.array(prev_layer.output_shape[self.out_dim-1:])).tolist()
        flatten_shape=prev_layer.output_shape[:self.out_dim-1]+(flatten_shape,)
        self.output_shape=flatten_shape
        Layer.connect(self, prev_layer)




    def __call__(self,layers):
        super(Flatten,self).__call__(layers)
        flatten_shape = np.prod(np.array(self.input_shape[self.out_dim - 1:])).tolist()
        flatten_shape = self.input_shape[:self.out_dim - 1] + (flatten_shape,)
        self.output_shape = flatten_shape

        return self



    def forward(self,is_training=True):
        inputs=self.input_tensor
        flatten_shape=inputs.shape[:self.out_dim-1]+(-1,)
        self.output_tensor=np.reshape(inputs,flatten_shape)
        if is_training:
            if self.require_grads:
                self.input_shape=inputs.shape
                self.grads = np.zeros_like(self.output_tensor)
        del inputs
        super().forward(is_training)




    def backward(self):
        for layer in self.inbound_layers:
            if layer.require_grads:
                layer.grads+=np.reshape(self.grads,self.input_shape)
            else:
                layer.grads=self.grads





class Dense(Layer):
    def __init__(self,n_out,n_in=None,initializer='Normal',activation='linear',kernel_regularizer=None):
        self.n_out=n_out
        self.n_in=n_in
        self.initializer = get_initializer(initializer)
        self.activator=get_activator(activation)
        self.kernel_regularizer=get_regularizer(kernel_regularizer)
        super(Dense,self).__init__()



    def connect(self,prev_layer):
        if prev_layer is None:
            assert self.n_in is not None
            assert self.input_shape is not None
        else:
            self.input_shape=prev_layer.output_shape

        self._initial_params()
        Layer.connect(self, prev_layer)
        self.output_shape = self.compute_output_shape()
        # W = Variable(self.initializer((n_in, self.n_out)))
        # b = Variable(np.zeros((1, self.n_out)))
        # W.grads = np.zeros_like(W.output_tensor) if W.require_grads else None
        # b.grads = np.zeros_like(b.output_tensor) if b.require_grads else None
        # self.variables.append(W)
        # self.variables.append(b)




    def __call__(self,prev_layer):
        super(Dense, self).__call__(prev_layer)
        self._initial_params()
        self.output_shape=self.compute_output_shape()
        return self



    def _initial_params(self):
        n_in = self.input_shape[-1]
        W = Variable(self.initializer((n_in, self.n_out)),name='dense_w')
        b = Variable(np.zeros((1, self.n_out)),name='dense_b')
        self.variables.append(W)
        self.variables.append(b)
        for var in self.variables:
            if var.require_grads:
                var.grads=np.zeros_like(var.output_tensor)



    def compute_output_shape(self):
        return self.input_shape[:-1]+(self.n_out,)





    def forward(self,is_training=True):
        W,b=self.variables
        output=self.input_tensor.dot(W.output_tensor)+b.output_tensor
        self.output_tensor=self.activator._forward(output,is_training=is_training)
        if is_training:
            if not W.require_grads:
                del self.input_tensor

            # if self.require_grads:
            #     self.grads = np.zeros_like(self.output_tensor)
        super().forward(is_training)





    def backward(self):
        W, b = self.variables
        grads=self.activator._backward(self.grads)
        if W.require_grads:
            W.grads+=self.input_tensor.T.dot(grads)
        if b.require_grads:
            b.grads+=np.sum(grads,axis=0,keepdims=True)
        self.timedist_grads=grads.dot(W.output_tensor.T)
        for layer in self.inbound_layers:
            if layer.require_grads:
                layer.grads+=self.timedist_grads
            else:
                self.grads=grads
                layer.grads=grads

        # self.counts-=1
        #
        # if self.counts==0:
        #     del self.inputs
        #     gc.collect()



