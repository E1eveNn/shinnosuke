from .Base import Layer
import numpy as np
from ..utils.Activator import get_activator



class Activation(Layer):
    def __init__(self,act_name='relu'):
        self.activator=get_activator(act_name)
        super(Activation,self).__init__()


    def connect(self,prev_layer):

        self.output_shape=prev_layer.output_shape
        Layer.connect(self, prev_layer)




    def __call__(self,prev_layer):
        super(Activation,self).__call__(prev_layer)
        self.output_shape=self.input_shape
        return self



    def forward(self,is_training=True):
        self.output_tensor=self.activator._forward(self.input_tensor,is_training)
        super().forward(is_training)
        if is_training:
            if self.require_grads:
                self.grads = np.zeros_like(self.output_tensor)



    def backward(self):
        for layer in self.inbound_layers:
            if layer.require_grads:
                layer.grads+=self.activator._backward(self.grads)
            else:
                layer.grads=self.grads