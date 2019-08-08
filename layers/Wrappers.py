from .Base import Layer
import numpy as np


class TimeDistributed(Layer):
    def __init__(self,layer,**kwargs):
        super(TimeDistributed,self).__init__(**kwargs)
        self.layer=layer


    def __call__(self, prev_layer):
        super(TimeDistributed,self).__call__(prev_layer)
        assert len(self.input_shape)==3
        self.layer.input_shape=self.input_shape
        self.layer._initial_params()
        self.output_shape=self.layer.compute_output_shape()
        self.layer.output_shape=self.output_shape
        self.variables=self.layer.variables
        return self


    def connect(self,inbound_layer):
        self.input_shape=inbound_layer.output_shape
        self.layer.input_shape = self.input_shape
        self.layer._initial_params()
        self.output_shape = self.layer.compute_output_shape(self.input_shape)
        self.layer.output_shape = self.output_shape
        self.variables = self.layer.variables
        Layer.connect(self, inbound_layer)




    def forward(self,is_training):
        inputs=self.input_tensor
        timesteps=self.input_shape[1]
        output=np.zeros((inputs.shape[0],timesteps,self.output_shape[2]))
        for t in range(timesteps):
            self.layer.input_tensor=self.input_tensor[:,t,:]
            self.layer.forward(is_training=is_training)
            output[:,t,:]=self.layer.output_tensor
        self.output_tensor=output
        super().forward(is_training)



    def backward(self):

        timesteps=self.output_shape[1]
        for t in range(timesteps):
            self.layer.input_tensor=self.input_tensor[:,t,:]
            self.layer.grads=self.grads[:,t,:]
            self.layer.backward()
            for layer in self.inbound_layers:
                if layer.require_grads:
                    layer.grads[:,t,:] += self.layer.timedist_grads
                else:
                    layer.grads = self.grads
