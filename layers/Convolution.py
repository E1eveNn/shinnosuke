from ..utils.Initializers import get_initializer
from ..utils.Activator import get_activator
from ..utils.ConvCol import im2col,col2im
from .Base import Layer,Variable
import numpy as np






class Conv2D(Layer):
    def __init__(self,filter_nums,filter_size,input_shape=None,stride=1,padding='VALID',activation='linear',initializer='Normal'):
        self.filter_nums=filter_nums
        self.filter_size=filter_size
        self.input_shape=input_shape
        self.stride=stride
        self.padding=padding
        self.activator = get_activator(activation)
        self.initializer=get_initializer(initializer)
        super(Conv2D,self).__init__(input_shape=input_shape)




    def connect(self,prev_layer):
        if prev_layer is None:
            assert self.input_shape is not None
            input_shape=self.input_shape

        else:
            input_shape=prev_layer.output_shape
            Layer.connect(self, prev_layer)


        assert len(input_shape)==4
        batch_nums,n_C_prev,n_H_prev,n_W_prev,=input_shape
        filter_h,filter_w=self.filter_size

        assert self.padding.__class__.__name__=='str'
        self.padding=self.padding.upper()
        if self.padding=='SAME':
            n_H=n_H_prev
            n_W=n_W_prev
            pad_h=(self.stride*(n_H_prev-1)-n_H_prev+filter_h)//2
            pad_w=(self.stride*(n_W_prev-1)-n_W_prev+filter_w)//2

            self.pad_size=(pad_h,pad_w)
        elif self.padding=='VALID':
            n_H=(n_H_prev-filter_h)//self.stride+1
            n_W=(n_W_prev-filter_w)//self.stride+1

            self.pad_size=(0,0)
        else:
            raise TypeError('Unknown padding type!plz inputs SAME or VALID')


        self.output_shape=(batch_nums,self.filter_nums,n_H,n_W)

        W = Variable(self.initializer((self.filter_nums, n_C_prev, filter_h, filter_w)))
        b = Variable(np.random.randn(1, self.filter_nums))
        W.grads = np.zeros_like(W.output_tensor) if W.require_grads else None
        b.grads = np.zeros_like(b.output_tensor) if b.require_grads else None
        self.variables.append(W)
        self.variables.append(b)


    def __call__(self,prev_layer):

        super(Conv2D, self).__call__(prev_layer)
        batch_nums, n_C_prev, n_H_prev, n_W_prev, = self.input_shape
        filter_h, filter_w = self.filter_size

        assert self.padding.__class__.__name__ == 'str'
        self.padding = self.padding.upper()
        if self.padding == 'SAME':
            n_H = n_H_prev
            n_W = n_W_prev
            pad_h = (self.stride * (n_H_prev - 1) - n_H_prev + filter_h) // 2
            pad_w = (self.stride * (n_W_prev - 1) - n_W_prev + filter_w) // 2

            self.pad_size = (pad_h, pad_w)
        elif self.padding == 'VALID':
            n_H = (n_H_prev - filter_h) // self.stride + 1
            n_W = (n_W_prev - filter_w) // self.stride + 1

            self.pad_size = (0, 0)
        else:
            raise TypeError('Unknown padding type!plz inputs SAME or VALID')

        self.output_shape = (batch_nums, self.filter_nums, n_H, n_W)
        W=Variable(self.initializer((self.filter_nums, n_C_prev, filter_h, filter_w)))
        b=Variable(np.random.randn(1, self.filter_nums))
        W.grads = np.zeros_like(W.output_tensor) if W.require_grads else None
        b.grads = np.zeros_like(b.output_tensor) if b.require_grads else None
        self.variables.append(W)
        self.variables.append(b)


        return self


    def _initial_params(self,*args):
        batch_nums, n_C_prev, n_H_prev, n_W_prev, = self.input_shape
        filter_h, filter_w = self.filter_size


        W = Variable(self.initializer((self.filter_nums, n_C_prev, filter_h, filter_w)))
        b = Variable(np.random.randn(1, self.filter_nums))

        self.variables.append(W)
        self.variables.append(b)
        for var in self.variables:
            if var.require_grads:
                var.grads=np.zeros_like(var.output_tensor)


    def forward(self,is_training=True):
        W,b=self.variables
        # pad
        inputs = np.pad(self.input_tensor, ((0, 0), (0, 0), self.pad_size, self.pad_size), 'constant')
        # padded size
        batch_nums = inputs.shape[0]
        _,n_C,n_H,n_W=self.output_shape

        col=im2col(inputs,self.output_shape,self.filter_size,self.stride)

        col_W=W.output_tensor.reshape(self.filter_nums,-1).T

        output=col.dot(col_W)+b.output_tensor
        output=output.reshape(batch_nums,n_H,n_W,-1).transpose(0,3,1,2)
        self.output_tensor=self.activator._forward(output)

        #store it for bp
        if is_training:
            self.input_shape=self.input_tensor.shape
            self.col = col
            self.col_W=col_W
            if self.require_grads:
                self.grads = np.zeros_like(self.output_tensor)
        del self.input_tensor
        super().forward(is_training)






    def backward(self):
        W,b=self.variables
        grads=self.activator._backward(self.grads)
        filter_nums,n_C,filter_h,filter_w=W.output_shape
        grads=grads.transpose(0,2,3,1).reshape(-1,filter_nums)
        if W.require_grads:
            dW = self.col.T.dot(grads)
            W.grads += dW.transpose(1, 0).reshape(filter_nums, n_C, filter_h, filter_w)
        if b.require_grads:
            b.grads += np.sum(grads,axis=0)
        for layer in self.inbounds:
            if layer.require_grads:
                dcol = grads.dot(self.col_W.T)

                layer.grads+=col2im(self.input_shape,self.pad_size,self.filter_size,self.stride,dcol)
            else:
                layer.grads=grads
        del self.output_tensor






class ZeroPadding2D(Layer):
    def __init__(self,pad_size):
        assert len(pad_size)==2
        self.pad_h,self.pad_w=pad_size
        self.require_grads=False
        super(ZeroPadding2D,self).__init__()



    def connect(self,prev_layer):
        (batch_nums,C,H,W)=prev_layer.output_shape
        self.output_shape = (batch_nums,C,H+2*self.pad_h,W+2*self.pad_w)
        Layer.connect(self, prev_layer)



    def __call__(self,prev_layer):
        super(ZeroPadding2D,self).__call__(prev_layer)
        (batch_nums, C, H, W) = self.input_shape
        self.output_shape = (batch_nums, C, H + 2 * self.pad_h, W + 2 * self.pad_w)
        return self



    def forward(self,is_training=True):
        inputs=self.input_tensor
        self.output_tensor = np.pad(inputs, ((0, 0), (0, 0), (self.pad_h,self.pad_h), (self.pad_w,self.pad_w)), 'constant')
        if is_training:
            if self.require_grads:
                self.grads=np.zeros_like(inputs)
        del inputs
        super().forward(is_training)





    def backward(self):
        for layer in self.inbounds:
            if layer.require_grads:
                layer.grads+=self.grads[:,:,self.pad_h:-self.pad_h,self.pad_w:-self.pad_w]
            else:
                layer.grads=self.grads





class MaxPooling2D(Layer):
    def __init__(self,pool_size,stride=None):
        assert len(pool_size)==2
        assert pool_size[0]==pool_size[1]
        self.pool_size=pool_size
        self.stride=pool_size[0] if stride is None else stride
        super(MaxPooling2D,self).__init__()



    def connect(self,prev_layer):
        batch_nums,n_C,n_H_prev,n_W_prev=prev_layer.output_shape
        pool_h,pool_w=self.pool_size
        n_H,n_W=(n_H_prev-pool_h)//self.stride+1,(n_W_prev-pool_w)//self.stride+1
        self.output_shape=(batch_nums,n_C,n_H,n_W)
        Layer.connect(self, prev_layer)


    def __call__(self,prev_layer):
        super(MaxPooling2D,self).__call__(prev_layer)
        batch_nums, n_C, n_H_prev, n_W_prev = self.input_shape
        pool_h, pool_w = self.pool_size
        n_H, n_W = (n_H_prev - pool_h) // self.stride + 1, (n_W_prev - pool_w) // self.stride + 1
        self.output_shape = (batch_nums, n_C, n_H, n_W)
        return self



    def forward(self,is_training=True):
        inputs=self.input_tensor
        _, n_C, n_H, n_W = self.output_shape
        batch_nums=inputs.shape[0]

        col = im2col(inputs,self.output_shape,self.pool_size,self.stride)
        pool_h,pool_w=self.pool_size
        col = col.reshape(-1, pool_h * pool_w)

        pool_argmax = np.argmax(col, axis=1)

        outputs = np.max(col, axis=1)
        self.output_tensor = outputs.reshape(batch_nums, n_H, n_W, n_C).transpose(0, 3, 1, 2)
        if is_training:
            self.inputs_shape = inputs.shape
            self.argmax_index = pool_argmax
            if self.require_grads:
                self.grads = np.zeros_like(self.output_tensor)
            del inputs
        super().forward(is_training)



    def backward(self):
        grads = self.grads.transpose(0, 2, 3, 1)
        pool_h,pool_w=self.pool_size
        dmax = np.zeros((grads.size, pool_h*pool_w))
        dmax[np.arange(self.argmax_index.size), self.argmax_index.flatten()] = grads.flatten()
        dmax = dmax.reshape(grads.shape + (pool_h*pool_w,))

        dcol = dmax.reshape(np.prod(dmax.shape[:3]), -1)
        outputs = col2im(self.inputs_shape,(0,0),self.pool_size, self.stride, dcol)
        for layer in self.inbounds:
            if layer.require_grads:
                layer.grads += outputs
            else:
                layer.grads = grads







class MeanPooling2D(Layer):
    def __init__(self,pool_size,stride=None):
        assert len(pool_size)==2
        assert pool_size[0]==pool_size[1]
        self.pool_size=pool_size
        self.stride=pool_size[0] if stride is None else stride
        super(MeanPooling2D,self).__init__()



    def connect(self,prev_layer):
        batch_nums,n_C,n_H_prev,n_W_prev=prev_layer.output_shape
        pool_h,pool_w=self.pool_size
        n_H,n_W=(n_H_prev-pool_h)//self.stride+1,(n_W_prev-pool_w)//self.stride+1
        self.output_shape=(batch_nums,n_C,n_H,n_W)
        Layer.connect(self, prev_layer)



    def __call__(self,prev_layer):
        super(MeanPooling2D,self).__call__(prev_layer)
        batch_nums, n_C, n_H_prev, n_W_prev = self.input_shape
        pool_h, pool_w = self.pool_size
        n_H, n_W = (n_H_prev - pool_h) // self.stride + 1, (n_W_prev - pool_w) // self.stride + 1
        self.output_shape = (batch_nums, n_C, n_H, n_W)
        return self



    def forward(self,is_training=True):
        inputs = self.input_tensor
        _, n_C, n_H, n_W = self.output_shape
        batch_nums=inputs.shape[0]

        col = im2col(inputs,self.output_shape,self.pool_size,self.stride)
        pool_h,pool_w=self.pool_size
        col = col.reshape(-1, pool_h * pool_w)

        pool_argmean = np.array([range(col.shape[1])])

        outputs = np.mean(col, axis=1)
        self.output_tensor = outputs.reshape(batch_nums, n_H, n_W, n_C).transpose(0, 3, 1, 2)
        if is_training:
            self.inputs_shape = inputs.shape
            self.argmean_index = pool_argmean
            if self.require_grads:
                self.grads = np.zeros_like(self.output_tensor.value)
            del inputs
        super().forward(is_training)



    def backward(self):
        grads = self.grads.transpose(0, 2, 3, 1)
        pool_h,pool_w=self.pool_size

        dmean=np.repeat(grads.flatten(),self.argmean_index.size)/(pool_w*pool_h)
        dmean = dmean.reshape(grads.shape + (pool_h*pool_w,))

        dcol = dmean.reshape(np.prod(dmean.shape[:3]), -1)
        outputs = col2im(self.inputs_shape,(0,0),self.pool_size, self.stride, dcol)

        for layer in self.inbounds:
            if layer.require_grads:
                layer.grads += outputs
            else:
                layer.grads = grads
        del self.output_tensor













