
import numpy as np





class Node(object):
    '''
    basic class for layers in network
    '''
    def __init__(self,outbound_layers=None,shape=None,output_tensor=None,name=None):
        '''
        :param inbound_edges:list,collections of all input edges
        '''

        self.inbounds=[]

        self.outbound_layers=[] if outbound_layers is None else list(outbound_layers)

        self.output_tensor=output_tensor
        #current node's name
        self.output_shape=shape

        self.require_grads=True
        #store the grads from next layer

        self.grads=None

        self.name=name



    def forward(self):
        pass


    def backward(self):
        pass


    def __add__(self, other):
        # return add(self, other)
        return Add()([self,other])

    def __sub__(self, other):
        return Add()([self,Negative()(other)])


    def __matmul__(self, other):
        return Matmul()([self, other])


    def __mul__(self, other):
        return Multiply()([self, other])


    def __neg__(self):
        return Negative()(self)



    def get_shape(self):
        return self.output_tensor.shape



    def get_value(self):
        return self.output_tensor



    def grad(self,cur_grad=None):
        if cur_grad is not  None:
            self.grads=np.asarray(cur_grad)
        out=self.__recursive_find_output(self)
        self.__recursive_grad(out)



    def __recursive_find_output(self,var):
        if var.outbound_layers:
            for out in var.outbound_layers:
                return self.__recursive_find_output(out)
        else:
            return var


    def __recursive_grad(self,out):
        if out is not self:
            if out.inbounds:
                out.backward()
                for var in out.inbounds:
                    self.__recursive_grad(var)




class Constant(Node):
    def __init__(self,output_tensor,name='constant'):
        Node.__init__(self)
        self.output_tensor=np.array(output_tensor)
        self.name=name




class Variable(Node):
    def __init__(self,initial_value=None,shape=None,name='variable'):
        Node.__init__(self,shape=shape)
        if initial_value is None:
            self.output_tensor=initial_value
        else:
            self.output_tensor=np.array(initial_value)
            self.output_shape=self.output_tensor.shape
        self.name=name






#for nodes
# def add(a,b,outputs=None):
#     if outputs is None:
#         return Variable(initial_output_tensor=a.output_tensor+b.output_tensor)
#     else:
#         outputs.output_tensor=a.output_tensor+b.output_tensor
#         return outputs
#
#
# def negative(a,outputs=None):
#     if outputs is None:
#         return Variable(initial_output_tensor=-a.output_tensor)
#     else:
#         outputs.output_tensor=-a.output_tensor
#         return outputs
#
#
# def subtract(a,b,outputs=None):
#     if outputs is None:
#         return Variable(initial_output_tensor=a.output_tensor - b.output_tensor)
#     else:
#         outputs.output_tensor = a.output_tensor - b.output_tensor
#         return outputs
#
#
#
# def multiply(a,b,outputs=None):
#     if outputs is None:
#         return Variable(initial_output_tensor=a.output_tensor * b.output_tensor)
#     else:
#         outputs.output_tensor = a.output_tensor * b.output_tensor
#         return outputs
#
#
#
#
# def matmul(a, b, outputs=None):
#     if outputs is None:
#         return Variable(initial_output_tensor=np.dot(a.output_tensor,b.output_tensor))
#     else:
#         outputs.output_tensor = np.dot(a.output_tensor, b.output_tensor)
#         return outputs




class Layer(object):
    def __init__(self,inbounds=None, outbound_layers=None,input_shape=None,output_shape=None,input_tensor=None,output_tensor=None,variables=None):

        self.inbounds = [] if inbounds is None else [inbounds]

        self.outbound_layers = [] if outbound_layers is None else [outbound_layers]

        self.input_shape=input_shape

        self.output_shape=output_shape

        self.input_tensor=input_tensor

        self.output_tensor=output_tensor

        self.variables=[] if variables is None else list(variables)
        #record how many other layers require this layer
        self.counts=0
        self.require_grads=True
        self.grads=None


    def __call__(self, inbound):
        # inbound_node.counts+=1
        if not isinstance(inbound,(list,tuple)):
            inbound=[inbound]
        for layer in inbound:
            self.inbounds.append(layer)
            self.input_shape=layer.output_shape
            layer.outbound_layers.append(self)
        # for var in self.variables:
        #     if var.require_grads:
        #         var.grads=np.zeros_like(var.output_tensor)



    def connect(self,inbound):
        if inbound is None:
            pass
        else:
            self.inbounds.append(inbound)
            self.input_shape = inbound.output_shape
            inbound.outbound_layers.append(self)
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
        return Add()([self, Negative()(other)])



    def __matmul__(self, other):
        return Matmul()([self, other])


    def __mul__(self, other):
        return Multiply()([self, other])


    def __neg__(self):
        return Negative()(self)


    def get_value(self):
        return self.output_tensor



    def feed(self,inputs,objects='inputs'):
        if isinstance(inputs,(Node,Layer)):
            inputs=inputs.output_tensor
        inputs=np.asarray(inputs)
        if objects=='inputs':
            self.input_tensor=inputs
        elif objects=='outputs':
            self.output_tensor=inputs
        elif objects[:-1]=='variable':
            self.variables[int(objects[-1])]=inputs
        elif objects=='grads':
            self.grads=inputs
        else:
            raise ValueError('unknown objects type,only support for - inputs/outputs/grads/variable*')






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
            self.inbounds.append(inbound)
        self.variables=self.inbounds
        return self


    def grad(self, cur_grad=None):
        if cur_grad is not None:
            self.grads = np.asarray(cur_grad)
        out = self.__recursive_find_output(self)
        self.__recursive_grad(out)

    def __recursive_find_output(self, var):
        if var.outbound_layers:
            for out in var.outbound_layers:
                return self.__recursive_find_output(out)
        else:
            return var


    def __recursive_grad(self, out):
        if out is not self:
            if out.inbounds:
                out.backward()
                for var in out.inbounds:
                    self.__recursive_grad(var)



#for layer
class Add(Operation):


    def __call__(self, inbounds):
        super(Add,self).__call__(inbounds)
        x,y=inbounds
        shape_a=x.output_tensor.shape
        shape_b=y.output_tensor.shape
        assert len(shape_a)==len(shape_b)
        output_shape=tuple()
        for a,b in zip(shape_a[1:],shape_b[1:]):
            output_shape+=(max(a,b),)
        output_shape=(None,)+output_shape
        self.output_shape=output_shape
        if x.output_tensor is not None and y.output_tensor is not None:
            self.output_tensor=x.output_tensor+y.output_tensor
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)
            if y.require_grads:
                y.grads=np.zeros_like(y.output_tensor)

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
        if x.require_grads:
            while grad_x.ndim >x.output_tensor.ndim:
                grad_x=np.sum(grad_x,axis=0)
            for axis,size in enumerate(x.output_tensor.shape):
                #in case of broadcast,for example, when forward propagation,x shape:(1,3,3,3),y shape:(3,3,3,3),x+y shape:(3,3,3,3),so grad shape does,and grad_x shape should be (1,3,3,3),thus needs to sum axis.
                if size==1:
                    grad_x=np.sum(grad_x,axis=axis,keepdims=True)
            x.grads += grad_x
        if y.require_grads:
            while grad_y.ndim >y.output_tensor.ndim:
                grad_y=np.sum(grad_y,axis=0)
            for axis,size in enumerate(y.output_tensor.shape):
                #in case of broadcast,for example, when forward propagation,x shape:(1,3,3,3),y shape:(3,3,3,3),x+y shape:(3,3,3,3),so grad shape does,and grad_x shape should be (1,3,3,3),thus needs to sum axis.
                if size==1:
                    grad_y=np.sum(grad_y,axis=axis,keepdims=True)

            y.grads+=grad_y






class Negative(Operation):
    def __call__(self, inbound):
        super(Negative,self).__call__([inbound])
        x=inbound
        self.output_shape=x.output_shape
        if x.output_tensor is not None:
            self.output_tensor=-x.output_tensor
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)

        return self



    def forward(self,is_training=True):
        x,=self.variables
        self.output_tensor=-x.output_tensor
        if is_training:
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)
        super().forward(is_training)




    def backward(self):
        x, = self.variables
        if x.require_grads:
            x.grads+=-self.grads



class Multiply(Operation):

    def __call__(self, inbounds):
        super(Multiply,self).__call__(inbounds)
        x,y=inbounds
        shape_a=x.output_tensor.shape
        shape_b=y.output_tensor.shape
        assert shape_a==shape_b
        self.output_shape=shape_a
        if x.output_tensor is not None and y.output_tensor is not None:
            self.output_tensor=x.output_tensor*y.output_tensor
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)
            if y.require_grads:
                y.grads=np.zeros_like(y.output_tensor)

        return self




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
        if x.require_grads:
            while grad_x.ndim > x.output_tensor.ndim:
                grad_x = np.sum(grad_x, axis=0)
            for axis, size in enumerate(x.output_tensor.shape):
                # in case of broadcast,for example, when forward propagation,x shape:(1,3,3,3),y shape:(3,3,3,3),x+y shape:(3,3,3,3),so grad shape does,and grad_x shape should be (1,3,3,3),thus needs to sum axis.
                if size == 1:
                    grad_x = np.sum(grad_x, axis=axis, keepdims=True)
            grad_x=grad_x*y.output_tensor
            x.grads += grad_x

        if y.require_grads:
            while grad_y.ndim > y.output_tensor.ndim:
                grad_y = np.sum(grad_y, axis=0)
            for axis, size in enumerate(y.output_tensor.shape):
                # in case of broadcast,for example, when forward propagation,x shape:(1,3,3,3),y shape:(3,3,3,3),x+y shape:(3,3,3,3),so grad shape does,and grad_x shape should be (1,3,3,3),thus needs to sum axis.
                if size == 1:
                    grad_y = np.sum(grad_y, axis=axis, keepdims=True)
            grad_y=grad_y*x.output_tensor


            y.grads += grad_y



class Matmul(Operation):
    def __call__(self, inbounds):
        super(Matmul,self).__call__(inbounds)
        x,y=inbounds
        shape_a=x.output_shape
        shape_b=y.output_shape
        assert shape_a[-1]==shape_b[0]
        self.output_shape=shape_a[:-1]+shape_b[1:]
        if x.output_tensor is not None and y.output_tensor is not None:
            self.output_tensor=x.output_tensor.dot(y.output_tensor)
            if x.require_grads:
                x.grads=np.zeros_like(x.output_tensor)
            if y.require_grads:
                y.grads=np.zeros_like(y.output_tensor)

        return self


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
        if x.require_grads:
            grad_x=np.dot(self.grads,y.output_tensor.T)
            x.grads += grad_x
        if y.require_grads:
            grad_y=np.dot(x.output_tensor.T,self.grads)
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
        if x.require_grads:
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
        if x.require_grads:
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
        if x.require_grads:
            x.grads += -self.grads*1/np.square(x.output_tensor)





