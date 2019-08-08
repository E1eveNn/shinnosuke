from ..utils.Initializers import get_initializer
from .Base import Layer,Variable
import numpy as np



class BatchNormalization(Layer):
    def __init__(self,epsilon=1e-3,momentum=0.9,axis=-1,gamma_initializer='ones',beta_initializer='zeros',moving_mean_initializer='zeros', moving_variance_initializer='ones'):
        # axis=-1 when input Fully Connected Layers(data shape:(M,N),where M donotes Batch-size,and N represents feature nums)
        # axis=1 when input Convolution Layers(data shape:(M,C,H,W),represents Batch-size,Channels,Height,Width,respectively)

        self.epsilon=epsilon
        self.axis=axis
        self.momentum=momentum
        self.gamma_initializer=get_initializer(gamma_initializer)
        self.beta_initializer=get_initializer(beta_initializer)
        self.moving_mean_initializer=get_initializer(moving_mean_initializer)
        self.moving_variance_initializer=get_initializer(moving_variance_initializer)
        super(BatchNormalization,self).__init__()



    def connect(self,prev_layer):
        n_in=prev_layer.output_shape[self.axis]
        gamma = Variable(self.gamma_initializer(n_in))
        beta = Variable(self.beta_initializer(n_in))
        gamma.grads = np.zeros_like(gamma.output_tensor) if gamma.require_grads else None
        beta.grads = np.zeros_like(beta.output_tensor) if beta.require_grads else None
        self.variables.append(gamma)
        self.variables.append(beta)
        self.moving_mean=self.moving_mean_initializer(n_in)
        self.moving_variance=self.moving_variance_initializer(n_in)
        self.output_shape=prev_layer.output_shape

        Layer.connect(self, prev_layer)



    def __call__(self,prev_layer):
        super(BatchNormalization,self).__call__(prev_layer)
        n_in = self.input_shape[self.axis]
        gamma = Variable(self.gamma_initializer(n_in))
        beta = Variable(self.beta_initializer(n_in))
        gamma.grads = np.zeros_like(gamma.output_tensor) if gamma.require_grads else None
        beta.grads = np.zeros_like(beta.output_tensor) if beta.require_grads else None
        self.variables.append(gamma)
        self.variables.append(beta)
        self.moving_mean = self.moving_mean_initializer(n_in)
        self.moving_variance = self.moving_variance_initializer(n_in)
        self.output_shape=self.input_shape
        return self


    def forward(self,is_training=True):
        inputs=self.input_tensor
        gamma,beta=self.variables
        outputs = np.zeros_like(inputs)
        if is_training:
            self.cache = []
            for k in range(inputs.shape[self.axis]):
            #calc mean
                mean=np.mean(inputs[:,k])
                #calc var
                var=np.var(inputs[:,k])
                #x minus u
                xmu=inputs[:,k]-mean
                sqrtvar=np.sqrt(var+self.epsilon)
                normalized_x=xmu/sqrtvar
                outputs[:,k]=gamma.output_tensor[k]*normalized_x+beta.output_tensor[k]
                self.cache.append((xmu,sqrtvar,normalized_x))
                self.moving_mean[k]=self.momentum*self.moving_mean[k]+(1-self.momentum)*mean
                self.moving_variance[k] = self.momentum * self.moving_variance[k] + (1 - self.momentum) * var

        else:
            for k in range(inputs.shape[self.axis]):
                std=np.sqrt(self.moving_variance[k]+self.epsilon)
                outputs[:,k]=(gamma.output_tensor[k]/std)*inputs[:,k]+(beta.output_tensor[k]-gamma.output_tensor[k]*self.moving_mean[k]/std)


        self.output_tensor=outputs
        if self.require_grads:
            self.grads = np.zeros_like(self.output_tensor)
        super().forward(is_training)


    def backward(self):
        grads=self.grads
        gamma,beta=self.variables
        outputs=np.zeros_like(grads)
        for k in range(grads.shape[self.axis]):
            xmu,sqrtvar,normalzied_x=self.cache[k]
            if beta.require_grads:
                beta.grads[k]+=np.sum(grads[:,k])
            if gamma.require_grads:
                gamma.grads[k]+=np.sum(grads[:,k]*normalzied_x)

            dnormalized_x=grads[:,k]*gamma.output_tensor[k]
            # equals to var^-3/2,where sqrtvar=var^1/2
            dvar=np.sum(np.power(-1./sqrtvar,3)*xmu*dnormalized_x*0.5)

            dmean=np.sum(-dnormalized_x/sqrtvar)-dvar*2*np.mean(xmu)
            m=np.prod(np.asarray(xmu.shape)).tolist()
            outputs[:,k]=dnormalized_x/sqrtvar+dvar*2*xmu/m+dmean/m
        for layer in self.inbound_layers:
            if layer.require_grads:
                layer.grads+=outputs
            else:
                layer.grads=grads







