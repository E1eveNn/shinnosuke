
import numpy as np
import copy




class Optimizer():
    def __init__(self,lr,decay):
        self.lr=lr
        self.decay=decay
        self.iterations=0


    def update(self,trainable_variables):
        self.iterations+=1







class StochasticGradientDescent(Optimizer):
    def __init__(self,lr=0.01,decay=0.0,*args,**kwargs):
        super(StochasticGradientDescent,self).__init__(lr,decay)


    def update(self,trainable_variables):
        for var in trainable_variables:
            var.output_tensor=var.output_tensor-self.lr*var.grads
            var.grads=np.zeros_like(var.output_tensor)

        super(StochasticGradientDescent,self).update(trainable_variables)




class Momentum(Optimizer):
    def __init__(self,lr=0.01,decay=0.0,rho=0.9,*args,**kwargs):
        self.rho=rho
        self.velocity=None

        super(Momentum,self).__init__(lr,decay)


    def update(self,trainable_variables):
        if self.velocity is None:
            #initialize
            self.velocity=[np.zeros_like(p.output_tensor) for p in trainable_variables]

        for i,(v,var) in enumerate(zip(self.velocity,trainable_variables)):
            v=self.rho*v+(1-self.rho)*var.grads
            var.output_tensor-=self.lr*v
            self.velocity[i]=v

        super(Momentum,self).update(trainable_variables)



class RMSprop(Optimizer):
    def __init__(self,lr=0.001,decay=0.0,rho=0.9,epsilon=1e-7,*args,**kwargs):
        self.rho=rho
        self.epsilon=epsilon
        self.ms=None

        super(RMSprop,self).__init__(lr,decay)


    def update(self,trainable_variables):
        if self.ms is None:
            #initialize
            self.ms=[np.zeros_like(p.output_tensor) for p in trainable_variables]

        for i,(s,var) in enumerate(zip(self.ms,trainable_variables)):
            new_s=self.rho*s+(1-self.rho)*np.square(var.grads)
            var.output_tensor-=self.lr*var.grads/np.sqrt(new_s+self.epsilon)
            self.ms[i]=new_s

        super(RMSprop,self).update(trainable_variables)


class AdaGrad(Optimizer):
    def __init__(self,lr=0.01,decay=0.0,epsilon=1e-7):
        super(AdaGrad,self).__init__(lr,decay)
        self.epsilon=epsilon
        self.ms=None


    def update(self,trainable_variables):
        if self.ms is None:
            self.ms=[np.zeros_like(g.grads) for g in trainable_variables]
        for i,(s,var)in enumerate(zip(self.ms,trainable_variables)):
            s+=np.power(var.grads,2)
            var.output_tensor-=self.lr*var.grads/np.sqrt(s+self.epsilon)
            self.ms[i]=s
        super(AdaGrad,self).update(trainable_variables)



class AdaDelta(Optimizer):
    def __init__(self,decay=0.0,lr=1.0,rho=0.95,epsilon=1e-7):
        super(AdaDelta,self).__init__(lr,decay)
        self.rho=rho
        self.epsilon=epsilon
        self.ms=None
        self.delta_x=None


    def update(self,trainable_variables):
        if self.ms is None:
            self.ms=[np.zeros_like(g.grads) for g in trainable_variables]
        if self.delta_x is None:
            self.delta_x=[np.zeros_like(g.grads) for g in trainable_variables]

        for i,(s,var,x) in enumerate(zip(self.ms,trainable_variables,self.delta_x)):
            s=self.rho*s+(1-self.rho)*np.power(var.grads,2)
            g_=np.sqrt((x+self.epsilon)/(s+self.epsilon))*var.grads
            var.output_tensor-=g_
            x=self.rho*x+(1-self.rho)*np.power(g_,2)
            self.ms[i]=s
            self.delta_x[i]=x
        super(AdaDelta,self).update(trainable_variables)




class Adam(Optimizer):
    def __init__(self,lr=0.001,decay=0.0,beta1=0.9,beta2=0.999,epsilon=1e-7,*args,**kwargs):
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.ms=None
        self.vs=None
        super(Adam,self).__init__(lr,decay)


    def update(self,trainable_variables):
        self.iterations+=1
        if self.ms is None:
            #initialize
            self.ms=[np.zeros_like(p.output_tensor) for p in trainable_variables]
        if self.vs is None:
            #initialize
            self.vs=[np.zeros_like(p.output_tensor) for p in trainable_variables]

        for i,(v,m,var) in enumerate(zip(self.vs,self.ms,trainable_variables)):
            v = self.beta1 * v + (1 - self.beta1) * var.grads
            m=self.beta2*m+(1-self.beta2)*np.square(var.grads)
            v_correct=v/(1-pow(self.beta1,self.iterations))
            m_correct=m/(1-pow(self.beta2,self.iterations))
            var.output_tensor-=self.lr*(v_correct/(np.sqrt(m_correct)+self.epsilon))

            self.ms[i]=m
            self.vs[i]=v

        super(Adam, self).update(trainable_variables)




def get_optimizer(optimizer):
    if optimizer.__class__.__name__=='str':
        optimizer=optimizer.lower()
        if optimizer=='sgd':
            return StochasticGradientDescent()
        elif optimizer=='adam':
            return Adam()
        elif optimizer=='rmsprop':
            return RMSprop()
        elif optimizer=='momentum':
            return Momentum()
        elif optimizer=='adagrad':
            return AdaGrad()
        elif optimizer=='adadelta':
            return AdaDelta()
    elif isinstance(optimizer,Optimizer):
        return copy.deepcopy(optimizer)
    else:
        raise ValueError('unknown optimizer type!')

