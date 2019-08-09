
import numpy as np





class Node(object):
    '''
    basic class for layers in network
    '''
    def __init__(self,outbound_layers=None,shape=None,output_tensor=None,name=None):
        '''
        :param inbound_edges:list,collections of all input edges
        '''

        self.inbound_layers=[]

        self.outbound_layers=[] if outbound_layers is None else list(outbound_layers)

        self.output_tensor=output_tensor
        #current node's name
        self.output_shape=shape

        self.require_grads=True
        #store the grads from next layer

        self.grads=0

        self.op=None

        self.name=name



    def forward(self):
        pass


    def backward(self):
        pass



    def __add__(self, other):
        outputs = add(self, other)
        self.outbound_layers.append(outputs)
        other.outbound_layers.append(outputs)
        outputs.inbound_layers.extend([self,other])
        outputs.op=Add
        return outputs



    def __sub__(self, other):
        outputs = subtract(self,other)
        self.outbound_layers.append(outputs)
        other.outbound_layers.append(outputs)
        outputs.inbound_layers.extend([self, other])
        return outputs


    def __matmul__(self, other):
        outputs = matmul(self, other)
        self.outbound_layers.append(outputs)
        other.outbound_layers.append(outputs)
        outputs.inbound_layers.extend([self,other])
        return outputs


    def __mul__(self, other):
        outputs = multiply(self, other)
        self.outbound_layers.append(outputs)
        other.outbound_layers.append(outputs)
        outputs.inbound_layers.extend([self,other])
        outputs.op=Multiply
        return outputs


    def __truediv__(self, other):
        outputs = truediv(self, other)
        self.outbound_layers.append(outputs)
        other.outbound_layers.append(outputs)
        outputs.inbound_layers.extend([self, other])
        return outputs




    def __neg__(self):
        outputs = negative(self)
        self.outbound_layers.append(outputs)
        outputs.inbound_layers.append(self)
        return outputs


    @property
    def get_value(self):
        return self.output_tensor



    def grad(self):
        if not self.outbound_layers:
            self.grads=1
            return self.grads
        





class Constant(Node):
    def __init__(self,output_tensor,name='constant'):
        Node.__init__(self)
        self.output_tensor=np.array(output_tensor)
        self.name=name





class Variable(Node):
    def __init__(self,initial_output_tensor=None,shape=None,name='variable'):
        Node.__init__(self,shape=shape)
        if initial_output_tensor is None:
            self.output_tensor=initial_output_tensor
        else:
            self.output_tensor=np.array(initial_output_tensor)
            self.output_shape=self.output_tensor.shape
        self.name=name




#for nodes
def add(a,b,outputs=None):
    if outputs is None:
        return Variable(initial_output_tensor=a.output_tensor+b.output_tensor)
    else:
        outputs.output_tensor=a.output_tensor+b.output_tensor
        return outputs


def negative(a,outputs=None):
    if outputs is None:
        return Variable(initial_output_tensor=-a.output_tensor)
    else:
        outputs.output_tensor=-a.output_tensor
        return outputs


def subtract(a,b,outputs=None):
    if outputs is None:
        return Variable(initial_output_tensor=a.output_tensor - b.output_tensor)
    else:
        outputs.output_tensor = a.output_tensor - b.output_tensor
        return outputs



def multiply(a,b,outputs=None):
    if outputs is None:
        return Variable(initial_output_tensor=a.output_tensor * b.output_tensor)
    else:
        outputs.output_tensor = a.output_tensor * b.output_tensor
        return outputs




def matmul(a, b, outputs=None):
    if outputs is None:
        return Variable(initial_output_tensor=np.dot(a.output_tensor,b.output_tensor))
    else:
        outputs.output_tensor = np.dot(a.output_tensor, b.output_tensor)
        return outputs





def truediv(a,b,outputs=None):
    if outputs is None:
        return Variable(initial_output_tensor=a.output_tensor / b.output_tensor)
    else:
        outputs.output_tensor = a.output_tensor / b.output_tensor
        return outputs









class Layer(object):
    def __init__(self,inbound_layers=None, outbound_layers=None,input_shape=None,output_shape=None,input_tensor=None,output_tensor=None,variables=None):

        self.inbound_layers = [] if inbound_layers is None else list(inbound_layers)

        self.outbound_layers = [] if outbound_layers is None else list(outbound_layers)

        self.input_shape=input_shape

        self.output_shape=output_shape

        self.input_tensor=input_tensor

        self.output_tensor=output_tensor

        self.variables=[] if variables is None else list(variables)
        #record how many other layers require this layer
        self.counts=0
        self.require_grads=True
        self.grads=None


    def __call__(self, inbound_layer):
        # inbound_node.counts+=1
        if not isinstance(inbound_layer,(list,tuple)):
            inbound_layer=[inbound_layer]
        for layer in inbound_layer:
            self.inbound_layers.append(layer)
            self.input_shape=layer.output_shape
            layer.outbound_layers.append(self)
        # for var in self.variables:
        #     if var.require_grads:
        #         var.grads=np.zeros_like(var.output_tensor)



    def connect(self,inbound_layer):
        if inbound_layer is None:
            pass
        else:
            self.inbound_layers.append(inbound_layer)
            self.input_shape = inbound_layer.output_shape
            inbound_layer.outbound_layers.append(self)
        # for var in self.variables:
        #     if var.require_grads:
        #         var.grads=np.zeros_like(var.output_tensor)


    def _initial_params(self,*args):
        pass


    def compute_output_shape(self,*args):
        pass


    def forward(self,is_training):
        for layer in self.outbound_layers:
            layer.input_tensor = self.output_tensor
        if is_training:
            if self.require_grads:
                self.grads=np.zeros_like(self.output_tensor)



    def backward(self):
        raise NotImplementedError


    def __add__(self, other):
        return Add()([self, other])


    def __sub__(self, other):
        return Add()([self, Negative(other)])



    def __matmul__(self, other):
        return Matmul()([self, other])


    def __mul__(self, other):
        return Multiply()([self, other])


    def __neg__(self):
        return Negative()(self)




class Input(Layer):
    def __init__(self,shape,value=None):
        super(Input,self).__init__()
        self.input_shape=shape
        self.output_shape=self.input_shape
        self.output_tensor=value
        self.require_grads=False



    def connect(self,prev_layer):
        if prev_layer is not None:
            self.input_shape=prev_layer.output_shape
            self.output_shape=self.input_shape




    def forward(self,is_training):
        self.output_tensor=self.input_tensor
        super(Input,self).forward(is_training)



    def backward(self):
        pass







class Operation(Layer):

    def __call__(self,inbounds):
        Layer.__init__(self)
        for inbound in inbounds:
            # inbound.counts+=1
            inbound.outbound_layers.append(self)
            self.inbound_layers.append(inbound)
        self.variables=self.inbound_layers
        return self




#for layer
class Add(Operation):


    def __call__(self, inbounds):
        super(Add,self).__call__(inbounds)
        x,y=inbounds
        shape_a=x.output_shape
        shape_b=y.output_shape
        assert len(shape_a)==len(shape_b)
        output_shape=tuple()
        for a,b in zip(shape_a[1:],shape_b[1:]):
            output_shape+=(max(a,b),)
        output_shape=(None,)+output_shape
        self.output_shape=output_shape
        return self





    def forward(self,is_training=True):
        x,y=self.variables
        self.output_tensor=np.add(x.output_tensor,y.output_tensor)
        if is_training:
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)
            if y.require_grads:
                y.grads=np.zeros_like(x.output_tensor)
        super().forward(is_training)




    def backward(self):
        x,y=[node for node in self.variables]
        grad_x,grad_y=self.grads,self.grads
        while grad_x.ndim >x.output_tensor.ndim:
            grad_x=np.sum(grad_x,axis=0)
        for axis,size in enumerate(x.output_tensor.shape):
            #in case of broadcast,for example, when forward propagation,x shape:(1,3,3,3),y shape:(3,3,3,3),x+y shape:(3,3,3,3),so grad shape does,and grad_x shape should be (1,3,3,3),thus needs to sum axis.
            if size==1:
                grad_x=np.sum(grad_x,axis=axis,keepdims=True)

        while grad_y.ndim >y.output_tensor.ndim:
            grad_y=np.sum(grad_y,axis=0)
        for axis,size in enumerate(y.output_tensor.shape):
            #in case of broadcast,for example, when forward propagation,x shape:(1,3,3,3),y shape:(3,3,3,3),x+y shape:(3,3,3,3),so grad shape does,and grad_x shape should be (1,3,3,3),thus needs to sum axis.
            if size==1:
                grad_y=np.sum(grad_y,axis=axis,keepdims=True)

        x.grads+=grad_x
        y.grads+=grad_y


    @staticmethod
    def _backward(grads,x,y):
        x.grads=grads
        y.grads=grads








class Negative(Operation):


    def forward(self,is_training=True):
        x,=self.variables
        self.output_tensor=-x.output_tensor
        if is_training:
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)
        super().forward(is_training)




    def backward(self):
        x, = self.variables
        x.grads+=-self.grads



class Multiply(Operation):


    def forward(self,is_training=True):
        x, y = self.variables
        self.output_tensor=np.multiply(x.output_tensor,y.output_tensor)
        if is_training:
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)
            if y.require_grads:
                y.grads=np.zeros_like(x.output_tensor)

        super().forward(is_training)


    def backward(self):
        x, y = [node for node in self.variables]
        grad_x,grad_y = self.grads,self.grads
        while grad_x.ndim > x.output_tensor.ndim:
            grad_x = np.sum(grad_x, axis=0)
        for axis, size in enumerate(x.output_tensor.shape):
            # in case of broadcast,for example, when forward propagation,x shape:(1,3,3,3),y shape:(3,3,3,3),x+y shape:(3,3,3,3),so grad shape does,and grad_x shape should be (1,3,3,3),thus needs to sum axis.
            if size == 1:
                grad_x = np.sum(grad_x, axis=axis, keepdims=True)
        grad_x=grad_x*y.output_tensor


        while grad_y.ndim > y.output_tensor.ndim:
            grad_y = np.sum(grad_y, axis=0)
        for axis, size in enumerate(y.output_tensor.shape):
            # in case of broadcast,for example, when forward propagation,x shape:(1,3,3,3),y shape:(3,3,3,3),x+y shape:(3,3,3,3),so grad shape does,and grad_x shape should be (1,3,3,3),thus needs to sum axis.
            if size == 1:
                grad_y = np.sum(grad_y, axis=axis, keepdims=True)
        grad_y=grad_y*x.output_tensor

        x.grads += grad_x
        y.grads += grad_y


    @staticmethod
    def _backward(grads,x,y):
        x.grads=grads*y.output_tensor
        y.grads=grads*x.output_tensor



class Matmul(Operation):


    def forward(self,is_training=True):
        x, y = self.variables
        self.output_tensor=np.dot(x.output_tensor,y.output_tensor)
        if is_training:
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)
            if y.require_grads:
                y.grads=np.zeros_like(x.output_tensor)

        super().forward(is_training)



    def backward(self):
        x, y = [node for node in self.variables]
        #for example ,x shape:(4,3),y shape:(3,2),x dot y shape:(4,2),so does grad shape.
        grad_x=np.dot(self.grads,y.output_tensor.T)
        grad_y=np.dot(x.output_tensor.T,self.grads)
        x.grads += grad_x
        y.grads += grad_y



class Log(Operation):

    def forward(self,is_training=True):
        x,=self.variables
        self.output_tensor=np.log(x.output_tensor)
        if is_training:
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)

        super().forward(is_training)


    def backward(self):
        x,=self.variables
        x.grads+=self.grads*1/x.output_tensor




class Exp(Operation):



    def forward(self,is_training=True):
        x, = self.variables
        self.output_tensor=np.exp(x.output_tensor)
        if is_training:
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)

        super().forward(is_training)


    def backward(self):
        x,=self.variables
        x.grads+=self.grads*self.output_tensor




class Reciprocal(Operation):


    def forward(self,is_training=True):
        x,=self.variables
        self.output_tensor= 1./x.output_tensor
        if is_training:
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)

        super().forward(is_training)


    def backward(self):
        x,=self.variables
        x.grads += -self.grads*1/np.square(x.output_tensor)




a=Variable(3)
b=Variable(5)
c=Variable(4)
d=Variable(6)
e=a*b+c*d
e.grad()
print(e.grads)


