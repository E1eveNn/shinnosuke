from .Base import Layer,Variable
from ..utils.Activator import get_activator
from ..utils.Initializers import get_initializer
from ..utils.Preprocess import concatenate
import numpy as np





class Recurrent(Layer):
    #Base class for Recurrent layer
    def __init__(self,cell,return_sequences=False,return_state=False,stateful=False,input_length=None,**kwargs):
        '''
        :param cell: A RNN object
        :param return_sequences: return all output sequences if true,else return output sequences' last output
        :param return_state:if true ,return last state
        :param stateful:if true，the sequences last state will be used as next sequences initial state
        :param input_length: input sequences' length
        :param kwargs:
        '''
        super(Recurrent,self).__init__(**kwargs)
        self.cell=cell
        self.return_sequences=return_sequences
        self.return_state=return_state
        self.stateful=stateful
        self.input_length=input_length


    def connect(self,inbound_layer):
        super().connect(inbound_layer)



    def __call__(self, prev_layer):
        assert len(prev_layer.output_shape)==3,'Only support batch input'

        super(Recurrent,self).__call__(prev_layer)








class SimpleRNNCell(Layer):
    #cell class for SimpleRNN
    def __init__(self,units,activation='tanh',initializer='glorotuniform',recurrent_initializer='orthogonal',**kwargs):
        super(SimpleRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activator_cls = get_activator(activation).__class__
        self.initializer = get_initializer(initializer)
        self.recurrent_initializer = get_initializer(recurrent_initializer)
        self.__first_initialize=True


    def _initial_params(self,n_in,n_out):
        variables=[]
        Wxa=Variable(self.initializer((n_in,n_out)),name='Wxa')
        Waa=Variable(self.recurrent_initializer((n_out,n_out)),name='Waa')
        ba=Variable(np.zeros((1,n_out)),name='ba')
        variables.append(Wxa)
        variables.append(Waa)
        variables.append(ba)
        for var in variables:
            if var.require_grads:
                var.grads=np.zeros_like(var.output_tensor)

        return variables






    def _forward(self,inputs,variables,is_training,stateful):
        batch_nums,timesteps,n_vec=inputs.shape
        Wxa,Waa,ba=variables
        if is_training:
            self.timesteps=timesteps
            if self.__first_initialize:
                #first intialize prev_a
                self.reset_state(shape=(batch_nums,timesteps+1,self.units))
                self.__first_initialize=False
            if stateful:
                self.prev_a[:,0,:]=self.prev_a[:,-1,:]
            else:
                self.reset_state(shape=(batch_nums,timesteps+1,self.units))
        else:
            self.reset_state(shape=(batch_nums,timesteps+1,self.units))
        self.activations=[self.activator_cls() for _ in range(timesteps)]

        for t in range(1,timesteps+1):
            self.prev_a[:,t,:]=self.activations[t-1]._forward(inputs[:,t-1,:].dot(Wxa.output_tensor)+self.prev_a[:,t-1,:].dot(Waa.output_tensor)+ba.output_tensor)
        if is_training:
            self.input_tensor=inputs

        return self.prev_a[:,1:,:]



    def _backward(self,pre_grad,variables,return_sequences):
        Wxa,Waa,ba=variables
        grad = np.zeros_like(self.input_tensor)
        if  return_sequences:
            da_next = np.zeros_like(self.prev_a[:, 0, :])

            for t in reversed(range(self.timesteps)):

                dz = self.activations[t]._backward(pre_grad[:,t,:]+da_next)
                if Waa.require_grads:
                    Waa.grads += np.dot(self.prev_a[:, t - 1, :].T, dz)
                if Wxa.require_grads:
                    Wxa.grads += np.dot(self.input_tensor[:, t, :].T, dz)
                if ba.require_grads:
                    ba.grads += dz
                da_next = np.dot(dz, Waa.output_tensor)

                grad[:, t, :] = np.dot(dz, Wxa.output_tensor.T)
        else:
            da=pre_grad
            for t in reversed(range(self.timesteps)):
                da = self.activations[t]._backward(da)
                if Waa.require_grads:
                    Waa.grads+=np.dot(self.prev_a[:,t-1,:].T,da)
                if Wxa.require_grads:
                    Wxa.grads+=np.dot(self.input_tensor[:,t,:].T,da)
                if ba.require_grads:
                    ba.grads+=da

                grad[:,t,:]=np.dot(da,Wxa.output_tensor.T)
                da=np.dot(da,Waa)

        return grad






    def reset_state(self,shape):
        self.prev_a=np.zeros(shape)






class SimpleRNN(Recurrent):
    # Fully-connected RNN
    def __init__(self,units,activation='tanh',initializer='glorotuniform',recurrent_initializer='orthogonal',return_sequences=False,return_state=False,stateful=False,**kwargs):
        cell=SimpleRNNCell(units=units,activation=activation,initializer=initializer,recurrent_initializer=recurrent_initializer)
        super(SimpleRNN,self).__init__(cell,return_sequences=return_sequences,return_state=return_state,stateful=stateful,**kwargs)



    def __call__(self, prev_layer):
        batch_nums, timesteps, n_vec = prev_layer.output_shape
        if self.return_sequences:
            self.output_shape=(batch_nums,timesteps,self.cell.units)
        else:
            self.output_shape=(batch_nums,self.cell.units)
        self.variables=self.cell._initial_params(n_vec,self.cell.units)

        super(SimpleRNN,self).__call__(prev_layer)
        return self


    def connect(self,inbound_layer):
        if inbound_layer is None:
            batch_nums, timesteps, n_vec = self.input_shape
        else:
            batch_nums, timesteps, n_vec = inbound_layer.output_shape
        if self.return_sequences:
            self.output_shape = (batch_nums, timesteps, self.cell.units)
        else:
            self.output_shape = (batch_nums, 1, self.cell.units)
        self.variables = self.cell._initial_params(n_vec, self.cell.units)
        super().connect(inbound_layer)





    def forward(self,is_training):
        variables=self.variables

        output=self.cell._forward(self.input_tensor,variables,is_training,self.stateful)
        # Way,by=variables[3:]
        #Fully connected
        # output=hidden_output.dot(Way.output_tensor)+by.output_tensor
        if self.return_sequences:
            self.output_tensor=output
        else:
            self.output_tensor=output[:,-1,:]
        if self.return_state:
            self.output_tensor=[self.output_tensor,self.cell.prev_a]


        super().forward(is_training)



    def backward(self):
        grad = self.cell._backward(self.grads, self.variables,self.return_sequences)
        for layer in self.inbounds:
            if layer.require_grads:
                layer.grads+=grad
            else:
                layer.grads=grad




class LSTMCell(Layer):
    # cell class for SimpleRNN
    def __init__(self, units, activation='tanh',recurrent_activation='sigmoid',initializer='glorotuniform', recurrent_initializer='orthogonal',unit_forget_bias=True, **kwargs):
        super(LSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activator_cls = get_activator(activation).__class__
        self.recurrent_activator_cls=get_activator(recurrent_activation).__class__
        self.initializer = get_initializer(initializer)
        self.recurrent_initializer = get_initializer(recurrent_initializer)
        self.unit_forget_bias=unit_forget_bias
        self.__first_initialize = True


    def _initial_params(self, n_in, n_out):
        #Wf_l means forget gate linear weight,Wf_r represents forget gate recurrent weight.
        variables = []
        #forget gate
        Wf_l = Variable(self.initializer((n_in,n_out)), name='Wf_l')
        Wf_r = Variable(self.recurrent_initializer((n_out, n_out)), name='Wf_r')
        #update gate
        Wu_l = Variable(self.initializer((n_in,n_out)), name='Wu_l')
        Wu_r = Variable(self.recurrent_initializer((n_out, n_out)), name='Wu_r')
        # update unit
        Wc_l = Variable(self.initializer((n_in,n_out)), name='Wc_l')
        Wc_r = Variable(self.recurrent_initializer((n_out, n_out)), name='Wc_r')
        # output gate
        Wo_l = Variable(self.initializer((n_in,n_out)), name='Wo_l')
        Wo_r = Variable(self.recurrent_initializer((n_out, n_out)), name='Wo_r')

        Wf=concatenate(Wf_r,Wf_l,axis=0,name='Wf')
        Wu=concatenate(Wu_r,Wu_l,axis=0,name='Wu')
        Wc=concatenate(Wc_r,Wc_l,axis=0,name='Wc')
        Wo=concatenate(Wo_r,Wo_l,axis=0,name='Wo')

        del Wf_r,Wf_l,Wu_r,Wu_l,Wc_r,Wc_l,Wo_r,Wo_l
        if self.unit_forget_bias:
            bf = Variable(np.ones((1, n_out)), name='bf')
        else:
            bf = Variable(np.zeros((1, n_out)), name='bf')
        bu=Variable(np.zeros((1, n_out)), name='bu')
        bc=Variable(np.zeros((1, n_out)), name='bc')
        bo=Variable(np.zeros((1, n_out)), name='bo')
        variables.extend([Wf,Wu,Wc,Wo,bf,bu,bc,bo])
        for var in variables:
            if var.require_grads:
                var.grads=np.zeros_like(var.output_tensor)
        return variables



    def _forward(self, inputs, variables, is_training, stateful):
        batch_nums, timesteps, n_vec = inputs.shape
        Wf,Wu,Wc,Wo,bf,bu,bc,bo = variables
        if is_training:
            self.timesteps = timesteps
            if self.__first_initialize:
                # first intialize prev_a
                self.reset_state(shape=(batch_nums, timesteps + 1, self.units))
                self.__first_initialize = False
            if stateful:
                self.prev_a[:, 0, :] = self.prev_a[:, -1, :]
                self.c[:,0,:]=self.c[:,-1,:]
            else:
                self.reset_state(shape=(batch_nums, timesteps + 1, self.units))
        else:
            self.reset_state(shape=(batch_nums, timesteps + 1, self.units))

        self.activations = [self.activator_cls() for _ in range(timesteps)]
        self.recurrent_activations = [self.recurrent_activator_cls() for _ in range(3*timesteps)]

        z=np.zeros((batch_nums,timesteps,n_vec+self.units))
        w = np.concatenate((Wf.output_tensor, Wu.output_tensor, Wc.output_tensor, Wo.output_tensor), axis=1)
        b = np.concatenate((bf.output_tensor, bu.output_tensor, bc.output_tensor, bo.output_tensor), axis=1)
        for t in range(1, timesteps + 1):
            zt=np.concatenate((self.prev_a[:,t-1,:],inputs[:,t-1,:]),axis=1)
            ot=zt.dot(w)+b
            f=ot[:,:self.units]
            u=ot[:,self.units:self.units*2]
            c_tilde=ot[:,self.units*2:self.units*3]
            o=ot[:,self.units*3:]
            self.tao_f[:, t-1, :] = self.recurrent_activations[3*(t-1)]._forward(f)
            self.tao_u[:, t-1, :] = self.recurrent_activations[3*(t - 1)+1]._forward(u)
            self.c_tilde[:, t-1, :] = self.activations[t - 1]._forward(c_tilde)
            self.c[:,t,:]=self.tao_f[:,t-1,:]*self.c[:,t-1,:]+self.tao_u[:, t-1, :]*self.c_tilde[:, t-1, :]
            self.tao_o[:,t-1,:]=self.recurrent_activations[3*(t-1)+2]._forward(o)
            self.prev_a[:,t,:]=self.tao_o[:,t-1,:]*np.tanh(self.c[:,t,:])
            z[:,t-1,:]=zt


        if is_training:
            self.input_tensor = z


        return self.prev_a[:, 1:, :]



    def _backward(self, pre_grad, variables, return_sequences):
        Wf,Wu,Wc,Wo,bf,bu,bc,bo = variables
        grad = np.zeros_like(self.input_tensor)
        grad=grad[:,:,self.units:]
        if return_sequences:
            da_next = np.zeros_like(self.prev_a[:, 0, :])
            dc_next=np.zeros_like(self.c[:,0,:])
            for t in reversed(range(self.timesteps)):
                da = pre_grad[:, t, :] + da_next
                dtao_o=da*np.tanh(self.c[:,t+1,:])
                do=self.recurrent_activations[3*(t+1)-1]._backward(dtao_o)
                if Wo.require_grads:
                    Wo.grads += np.dot(self.input_tensor[:,t,:].T,do)
                if bo.require_grads:
                    bo.grads+=np.sum(do,axis=0,keepdims=True)
                #tanh backward
                dc=dc_next
                dc+=da*self.tao_o[:,t,:]*(1-np.square(np.tanh(self.c[:,t+1,:])))
                # dc=Tanh._backward(da*self.tao_o[:,t,:],self.c[:,t,:])+dc_next
                dc_tilde=dc*self.tao_u[:,t,:]
                dc_tilde_before_act=self.activations[t]._backward(dc_tilde)

                if Wc.require_grads:
                    Wc.grads += np.dot(self.input_tensor[:, t, :].T, dc_tilde_before_act)
                if bc.require_grads:
                    bc.grads += np.sum(dc_tilde_before_act,axis=0,keepdims=True)
                dtao_u=dc*self.c_tilde[:,t,:]
                du=self.recurrent_activations[3*(t+1)-2]._backward(dtao_u)
                if Wu.require_grads:
                    Wu.grads += np.dot(self.input_tensor[:, t, :].T, du)
                if bu.require_grads:
                    bu.grads += np.sum(du,axis=0,keepdims=True)
                dtao_f=dc*self.c[:,t,:]
                df=self.recurrent_activations[3*(t+1)-3]._backward(dtao_f)
                if Wf.require_grads:
                    Wf.grads += np.dot(self.input_tensor[:, t, :].T, df)
                if bf.require_grads:
                    bf.grads += np.sum(df,axis=0,keepdims=True)
                dz=df.dot(Wf.output_tensor.T)+du.dot(Wu.output_tensor.T)+do.dot(Wo.output_tensor.T)+dc_tilde_before_act.dot(Wc.output_tensor.T)

                da_next = dz[:,:self.units]
                dc_next = dc*self.tao_f[:,t,:]

                grad[:, t, :] = dz[:,self.units:]
        else:
            da_next = np.zeros_like(self.prev_a[:, 0, :])
            dc_next = np.zeros_like(self.c[:, 0, :])
            da = pre_grad + da_next
            for t in reversed(range(self.timesteps)):

                dtao_o = da * np.tanh(self.c[:, t+1, :])
                do = self.recurrent_activations[3*(t+1)-1]._backward(dtao_o)
                if Wo.require_grads:
                    Wo.grads += np.dot(self.input_tensor[:,t,:].T,do)
                if bo.require_grads:
                    bo.grads+=np.sum(do,axis=0,keepdims=True)
                dc = dc_next
                dc += da * self.tao_o[:, t, :] * (1 - np.square(np.tanh(self.c[:, t+1, :])))
                # dc = Tanh._backward(da * self.tao_o[:, t, :], np.tanh(self.c[:, t, :])) + dc_next
                dc_tilde = dc * self.tao_u[:, t, :]
                dc_tilde_before_act=self.activations[t]._backward(dc_tilde)

                if Wc.require_grads:
                    Wc.grads += np.dot(self.input_tensor[:, t, :].T, dc_tilde_before_act)
                if bc.require_grads:
                    bc.grads += np.sum(dc_tilde_before_act,axis=0,keepdims=True)
                dtao_u = dc * self.c_tilde[:, t, :]
                du = self.recurrent_activations[3*(t+1)-2]._backward(dtao_u)
                if Wu.require_grads:
                    Wu.grads += np.dot(self.input_tensor[:, t, :].T, du)
                if bu.require_grads:
                    bu.grads += np.sum(du,axis=0,keepdims=True)
                dtao_f = dc * self.c[:, t, :]
                df = self.recurrent_activations[3*(t+1)-3]._backward(dtao_f)
                if Wf.require_grads:
                    Wf.grads += np.dot(self.input_tensor[:, t, :].T, df)
                if bf.require_grads:
                    bf.grads += np.sum(df,axis=0,keepdims=True)

                dz = df.dot(Wf.output_tensor.T) + du.dot(Wu.output_tensor.T) + do.dot(Wo.output_tensor.T) + dc_tilde_before_act.dot(Wc.output_tensor.T)

                da = dz[:, :self.units]
                dc_next = dc * self.tao_f[:, t, :]

                grad[:, t, :] = dz[:, self.units:]


        return grad


    def reset_state(self, shape):
        #timesteps here equals to real timesteps+1
        batch_nums,timesteps,units=shape
        self.prev_a = np.zeros(shape)
        self.c = np.zeros(shape)
        self.tao_f=np.zeros((batch_nums,timesteps-1,units))
        self.tao_u=np.zeros((batch_nums,timesteps-1,units))
        self.tao_o=np.zeros((batch_nums,timesteps-1,units))
        self.c_tilde=np.zeros((batch_nums,timesteps-1,units))




class LSTM(Recurrent):
    # Fully-connected RNN
    def __init__(self, units, activation='tanh',recurrent_activation='sigmoid',initializer='glorotuniform', recurrent_initializer='orthogonal',unit_forget_bias=True,return_sequences=False, return_state=False, stateful=False, **kwargs):
        '''
        :param units: hidden unit nums
        :param activation:  update unit activation
        :param recurrent_activation: forget gate,update gate,and output gate activation
        :param initializer: same to activation
        :param recurrent_initializer:same to recurrent_activation
        :param unit_forget_bias:if True,add one to the forget gate bias,and force bias initialize as zeros
        :param return_sequences: return sequences or last output
        :param return_state: if True ,return output and last state
        :param stateful: same as SimpleRNN
        :param kwargs:
        '''
        cell = LSTMCell(units=units,activation=activation,recurrent_activation=recurrent_activation,initializer=initializer,recurrent_initializer=recurrent_initializer,unit_forget_bias=unit_forget_bias)
        super(LSTM, self).__init__(cell, return_sequences=return_sequences, return_state=return_state,stateful=stateful, **kwargs)



    def __call__(self, prev_layer):
        batch_nums, timesteps, n_vec = prev_layer.output_shape
        if self.return_sequences:
            self.output_shape = (batch_nums, timesteps, self.cell.units)
        else:
            self.output_shape = (batch_nums, self.cell.units)

        self.variables = self.cell._initial_params(n_vec, self.cell.units)


        super(LSTM, self).__call__(prev_layer)
        return self



    def connect(self, inbound_layer):
        if inbound_layer is None:
            batch_nums, timesteps, n_vec = self.input_shape
        else:
            batch_nums, timesteps, n_vec = inbound_layer.output_shape
        if self.return_sequences:
            self.output_shape = (batch_nums, timesteps, self.cell.units)
        else:self.output_shape = (batch_nums, 1, self.cell.units)
        self.variables = self.cell._initial_params(n_vec, self.cell.units)
        super().connect(inbound_layer)



    def forward(self, is_training):
        variables = self.variables

        output = self.cell._forward(self.input_tensor, variables, is_training, self.stateful)

        if self.return_sequences:
            self.output_tensor = output
        else:
            self.output_tensor = output[:, -1, :]
        if self.return_state:
            self.output_tensor = [self.output_tensor, self.cell.prev_a]

        super().forward(is_training)




    def backward(self):
        grad = self.cell._backward(self.grads, self.variables, self.return_sequences)
        for layer in self.inbounds:
            if layer.require_grads:
                layer.grads += grad
            else:
                layer.grads = grad




class GRUCell(Layer):
    # cell class for SimpleRNN
    def __init__(self, units, activation='tanh',recurrent_activation='sigmoid',initializer='glorotuniform', recurrent_initializer='orthogonal', **kwargs):
        super(GRUCell, self).__init__(**kwargs)
        self.units = units
        self.activator_cls = get_activator(activation).__class__
        self.recurrent_activator_cls=get_activator(recurrent_activation).__class__
        self.initializer = get_initializer(initializer)
        self.recurrent_initializer = get_initializer(recurrent_initializer)

        self.__first_initialize = True


    def _initial_params(self, n_in, n_out):
        #Wf_l means forget gate linear weight,Wf_r represents forget gate recurrent weight.
        variables = []

        Wc_l = Variable(self.initializer((n_in,n_out)), name='Wc_l')
        Wc_r = Variable(self.recurrent_initializer((n_out, n_out)), name='Wc_r')

        Wu_l = Variable(self.initializer((n_in,n_out)), name='Wu_l')
        Wu_r = Variable(self.recurrent_initializer((n_out, n_out)), name='Wu_r')

        Wr_l = Variable(self.initializer((n_in,n_out)), name='Wr_l')
        Wr_r = Variable(self.recurrent_initializer((n_out, n_out)), name='Wr_r')


        Wc=concatenate(Wc_r,Wc_l,axis=0,name='Wc')
        Wu=concatenate(Wu_r,Wu_l,axis=0,name='Wu')
        Wr=concatenate(Wr_r,Wr_l,axis=0,name='Wr')

        del Wr_r,Wr_l,Wu_r,Wu_l,Wc_r,Wc_l,

        bc = Variable(np.ones((1, n_out)), name='bc')


        bu=Variable(np.zeros((1, n_out)), name='bu')

        br=Variable(np.zeros((1, n_out)), name='br')
        variables.extend([Wc,Wu,Wr,bc,bu,br])

        return variables



    def _forward(self, inputs, variables, is_training, stateful):
        batch_nums, timesteps, n_vec = inputs.shape
        Wc,Wu,Wr,bc,bu,br = variables
        if is_training:
            self.timesteps = timesteps
            if self.__first_initialize:
                # first intialize prev_a
                self.reset_state(shape=(batch_nums, timesteps + 1, self.units))
                self.__first_initialize = False
            if stateful:

                self.c[:,0,:]=self.c[:,-1,:]
            else:
                self.reset_state(shape=(batch_nums, timesteps + 1, self.units))
        else:
            self.reset_state(shape=(batch_nums, timesteps + 1, self.units))

        self.activations = [self.activator_cls() for _ in range(timesteps)]
        self.recurrent_activations = [self.recurrent_activator_cls() for _ in range(2*timesteps)]


        w = np.concatenate((Wu.output_tensor, Wr.output_tensor), axis=1)
        b = np.concatenate((bu.output_tensor, br.output_tensor), axis=1)
        for t in range(1, timesteps + 1):
            c_tilde=np.concatenate((self.tao_r[:,t-1,:]*self.c[:,t-1,:],inputs[:,t-1,:]),axis=1).dot(Wc.output_tensor)+bc.output_tensor
            self.c_tilde[:, t - 1, :] = self.activations[t - 1]._forward(c_tilde)
            zt=np.concatenate((self.c[:,t-1,:],inputs[:,t-1,:]),axis=1)
            ot=zt.dot(w)+b
            u=ot[:,:self.units]
            r=ot[:,self.units:self.units*2]
            self.tao_u[:, t-1, :] = self.recurrent_activations[2*(t - 1)]._forward(u)
            self.tao_r[:, t - 1, :] = self.recurrent_activations[2 * (t - 1) + 1]._forward(r)
            self.c[:,t,:]=self.tao_u[:,t-1,:]*self.c_tilde[:,t-1,:]+(1-self.tao_u[:, t-1, :])*self.c[:, t-1, :]



        # if is_training:
        #     self.input_tensor = z


        return self.c[:, 1:, :]



    def _backward(self, pre_grad, variables, return_sequences):
        raise NotImplemented


    def reset_state(self, shape):
        #timesteps here equals to real timesteps+1
        batch_nums,timesteps,units=shape
        self.c = np.zeros(shape)
        self.c_tilde=np.zeros((batch_nums,timesteps-1,units))
        self.tao_u=np.zeros((batch_nums,timesteps-1,units))
        self.tao_r=np.zeros((batch_nums,timesteps-1,units))





class GRU(Recurrent):
    # Fully-connected RNN
    def __init__(self, units, activation='tanh',recurrent_activation='sigmoid',initializer='glorotuniform', recurrent_initializer='orthogonal',return_sequences=False, return_state=False, stateful=False, **kwargs):
        '''
        :param units: hidden unit nums
        :param activation:  update unit activation
        :param recurrent_activation: forget gate,update gate,and output gate activation
        :param initializer: same to activation
        :param recurrent_initializer:same to recurrent_activation
        :param unit_forget_bias:if True,add one to the forget gate bias,and force bias initialize as zeros
        :param return_sequences: return sequences or last output
        :param return_state: if True ,return output and last state
        :param stateful: same as SimpleRNN
        :param kwargs:
        '''
        cell = GRUCell(units=units,activation=activation,recurrent_activation=recurrent_activation,initializer=initializer,recurrent_initializer=recurrent_initializer)
        super(GRU, self).__init__(cell, return_sequences=return_sequences, return_state=return_state,stateful=stateful, **kwargs)



    def __call__(self, prev_layer):
        batch_nums, timesteps, n_vec = prev_layer.output_shape
        if self.return_sequences:
            self.output_shape = (batch_nums, timesteps, self.cell.units)
        else:
            self.output_shape = (batch_nums, self.cell.units)

        self.variables = self.cell._initial_params(n_vec, self.cell.units)


        super(GRU, self).__call__(prev_layer)
        return self



    def connect(self, inbound_layer):
        if inbound_layer is None:
            batch_nums, timesteps, n_vec = self.input_shape
        else:
            batch_nums, timesteps, n_vec = inbound_layer.output_shape
        if self.return_sequences:
            self.output_shape = (batch_nums, timesteps, self.cell.units)
        else:self.output_shape = (batch_nums, 1, self.cell.units)
        self.variables = self.cell._initial_params(n_vec, self.cell.units)
        super().connect(inbound_layer)



    def forward(self, is_training):
        variables = self.variables

        output = self.cell._forward(self.input_tensor, variables, is_training, self.stateful)

        if self.return_sequences:
            self.output_tensor = output
        else:
            self.output_tensor = output[:, -1, :]
        if self.return_state:
            self.output_tensor = [self.output_tensor, self.cell.prev_a]

        super().forward(is_training)




    def backward(self):
        grad = self.cell._backward(self.grads, self.variables, self.return_sequences)
        for layer in self.inbounds:
            if layer.require_grads:
                layer.grads += grad
            else:
                layer.grads = grad
