from .utils.Objectives import get_objective
from .utils.Optimizers import get_optimizer
from .utils.MiniBatch import get_batches
import time
import matplotlib.pyplot as plt
import os
import pickle




class Sequential():
    def __init__(self,layers=None):
        self.layers=[] if layers is None else layers
        self.train_loss=[]
        self.train_acc=[]
        self.valid_loss=[]
        self.valid_acc=[]
        self.process_bar_nums=30
        self.process_bar_trained='='
        self.process_bar_untrain='*'



    def add(self,layer):
        self.layers.append(layer)



    def compile(self,optimizer,loss):
        assert self.layers
        trainable_variables=[]
        # self.layers[0].first_layer=True
        next_layer=None
        for layer in self.layers:
            layer.connect(next_layer)
            next_layer=layer
            for var in layer.variables:
                if var.require_grads:
                    trainable_variables.append(var)
        self.trainable_variables=trainable_variables
        self.loss=get_objective(loss)
        self.optimizer=get_optimizer(optimizer)



    def fit(self,X,Y,batch_size=64,epochs=20,shuffle=True,validation_data=None,validation_ratio=0.1,draw_acc_loss=False,draw_save_path=None):

        if validation_data is None:
            if 0.<validation_ratio<1.:
                split=int(X.shape[0]*validation_ratio)
                valid_X,valid_Y=X[-split:],Y[-split:]
                train_X,train_Y=X[:-split],Y[:-split]
                validation_data=(valid_X,valid_Y)
            else:
                train_X, train_Y = X, Y
        else:
            valid_X, valid_Y=validation_data
            train_X,train_Y=X,Y


        for epoch in range(epochs):
            mini_batches=get_batches(train_X,train_Y,batch_size,epoch,shuffle)
            batch_nums=len(mini_batches)
            training_size=train_X.shape[0]
            batch_count=0
            trained_nums=0
            print('\033[0;31m Epoch[%d/%d]' % (epoch + 1, epochs))
            start_time = time.time()
            for xs,ys in mini_batches:

                batch_count+=1
                trained_nums+=xs.shape[0]
                #forward
                y_hat=self.predict(xs)


                #backward

                self.layers[-1].grads = self.loss.backward(y_hat, ys)

                for layer in reversed(self.layers):
                    layer.backward()

                end_time = time.time()
                gap = end_time - start_time

                self.optimizer.update(self.trainable_variables)

                batch_acc, batch_loss = self.__evaluate(y_hat, ys)

                self.train_loss.append(batch_loss)
                self.train_acc.append(batch_acc)
                if validation_data is not None:
                    valid_acc, valid_loss = self.evaluate(valid_X, valid_Y,batch_size=batch_size)
                    self.valid_loss.append(valid_loss)
                    self.valid_acc.append(valid_acc)



                if draw_acc_loss:
                    if len(self.train_loss)==2:
                        plt.ion()
                        plt.figure(figsize=(6, 7))
                        plt.title('batch-size='+str(batch_size)+',Epochs='+str(epochs))
                        ax1 = plt.subplot(2, 1, 1)
                        ax2 = plt.subplot(2, 1, 2)
                    if len(self.train_loss)>1:
                        self.draw_training(ax1,ax2,draw_save_path,epoch)


                trained_process_bar_nums=batch_count*self.process_bar_nums//batch_nums
                process_bar=self.process_bar_trained*trained_process_bar_nums+'>'+self.process_bar_untrain*(self.process_bar_nums-trained_process_bar_nums-1)
                if validation_data is not None:
                    print(
                        '\r{:d}/{:d} [{}] -{:.0f}s -{:.0f}ms/batch -batch_loss: {:.4f} -batch_acc: {:.4f} -val_loss: {:.4f} -val_acc: {:.4f}'.format(trained_nums, training_size, process_bar, gap, gap*1000 / batch_count,batch_loss, batch_acc, valid_loss, valid_acc), end='')
                else:
                    print('\r{:d}/{:d} [{}] -{:.0f}s -{:.0f}ms/batch -batch_loss: {:.4f} -batch_acc: {:.4f} '.format( trained_nums, training_size, process_bar, gap, gap * 1000 / batch_count,batch_loss, batch_acc), end='')
            print()




    def predict(self,X,is_training=True):
        self.layers[0].input_tensor=X
        for layer in self.layers:
            layer.forward(is_training=is_training)
        y_hat=self.layers[-1].output_tensor
        return y_hat



    def __evaluate(self,y_hat,y_true):
        acc = self.loss.calc_acc(y_hat,y_true)
        base_loss = self.loss.calc_loss(y_hat, y_true)

        return acc,base_loss






    def evaluate(self, X, Y, batch_size=None):
        if batch_size is not None:
            assert type(batch_size) is int
            ep = 0
            acc_list = []
            loss_list = []
            data_nums = X.shape[0]
            while True:
                sp = ep
                ep = min(sp + batch_size, data_nums)
                y_hat = self.predict(X[sp:ep], is_training=False)
                acc = self.loss.calc_acc(y_hat,Y[sp:ep])
                acc_list.append(acc)
                base_loss = self.loss.calc_loss(y_hat, Y[sp:ep])
                loss_list.append(base_loss)

                if ep == data_nums:
                    acc = sum(acc_list) / len(acc_list)
                    base_loss = sum(loss_list) / len(loss_list)
                    break
        else:
            y_hat = self.predict(X, is_training=False)
            acc = self.loss.calc_acc(y_hat,Y)
            base_loss = self.loss.calc_loss(y_hat, Y)
            regular_loss = 0
            # for layer in self.layers:
            #     regular_loss+=layer.add_loss
        return acc, base_loss



    def draw_training(self,ax1,ax2,draw_save_path,epoch):
        leg1=ax1.get_legend()
        ax1.plot(self.train_loss, color='blue', label='train')
        if self.valid_loss:
            ax1.plot(self.valid_loss, color='green', label='validation')
        ax1.set_xlabel('iter')
        ax1.set_ylabel('loss')
        if leg1 is None:
            ax1.legend(loc='best')
        leg2 = ax2.get_legend()
        ax2.plot(self.train_acc, color='red', label='train')
        if self.valid_acc:
            ax2.plot(self.valid_acc, color='yellow', label='validation')
        ax2.set_xlabel('iter')
        ax2.set_ylabel('acc')
        if leg2 is None:
            ax2.legend(loc='best')
        plt.pause(0.1)
        if draw_save_path is not None:
            assert draw_save_path.__class__.__name__=='str'
            draw_save_path=os.path.abspath(draw_save_path+'\\Epoch'+str(epoch))
            plt.savefig(draw_save_path,dpi=300)



    def pop(self,index=-1):
        layer=self.layers.pop(index)
        del layer
        print('success delete %s layer'%(layer.__class__.__name__))




    def save(self,save_path):
        with open(save_path+'.pkl','wb') as f:
            pickle.dump([self.layers,self.optimizer,self.loss],f)



    def load(self,model_path):
        with open(model_path + '.pkl', 'rb') as f:
            layers,optimizer,loss = pickle.load(f)

        self.layers=layers
        self.optimizer=optimizer
        self.loss=loss




    def __str__(self):
        bar_nums = 75
        print('*' * bar_nums)

        print('Layer(type)'.ljust(20),'Output Shape'.ljust(20) ,'Param'.ljust(12),'Connected to'.ljust(15))
        print('#' * bar_nums)
        total_params = 0
        for layer in self.layers:

            if layer.name is not None:
                layer_name = '%s (%s)'%(layer.name,layer.__class__.__name__)
            else:
                layer_name = str(layer.__class__.__name__)

            params = layer.params_count()
            total_params += params
            first = True
            if layer.inbounds:

                for prev_layer in layer.inbounds:
                    if prev_layer.name is not None:
                        connected = prev_layer.name
                    else:
                        connected = prev_layer.__class__.__name__
                    if first:
                        print(layer_name.ljust(20),str(layer.output_shape).ljust(20), str(params).ljust(12),connected.ljust(15))
                        first = False
                    else:
                        print(''.ljust(20),''.ljust(20), ''.ljust(12),connected.ljust(15))
            else:
                connected = '\n'
                print(layer_name.ljust(20),str(layer.output_shape).ljust(20), str(params).ljust(12),connected.ljust(15))
            print('-' * bar_nums)

        print('*' * bar_nums)
        trainable_params = 0
        for v in self.trainable_variables:
            trainable_params += v.output_tensor.size
        params_details = 'Total params: %d\n'%(total_params)
        params_details += 'Trainable params: %d\n'%(trainable_params)
        params_details += 'Non-trainable params: %d\n' % (total_params-trainable_params)
        return params_details





class Model():
    def __init__(self, inputs=None,outputs=None):
        self.inputs=inputs
        self.outputs=outputs
        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []
        self.process_bar_nums = 30
        self.process_bar_trained = '='
        self.process_bar_untrain = '*'




    def topological_sort(self,input_layers,mode='forward'):
        """
        Sort generic nodes in topological order using Kahn's Algorithm.

        `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

        Returns a list of sorted nodes.
        """
        G = {}
        graph = []
        if mode=='forward':
            trainable_variables=[]
            layers = [input_layers]
            while len(layers) > 0:
                n = layers.pop(0)
                if n not in G:
                    G[n] = {'in': set(), 'out': set()}
                for m in n.outbound_layers:
                    for var in m.variables:
                        if var.require_grads and var not in trainable_variables:
                            trainable_variables.append(var)
                    if m not in G:
                        G[m] = {'in': set(), 'out': set()}
                    G[n]['out'].add(m)
                    G[m]['in'].add(n)
                    layers.append(m)


            S = set([input_layers])
            while len(S) > 0:
                n = S.pop()

                graph.append(n)
                for m in n.outbound_layers:
                    G[n]['out'].remove(m)
                    G[m]['in'].remove(n)
                    # if no other incoming edges add to S
                    if len(G[m]['in']) == 0:
                        S.add(m)
            return graph, trainable_variables

        elif mode=='backward':

            layers = [input_layers]
            while len(layers) > 0:
                n = layers.pop(0)
                if n not in G:
                    G[n] = {'in': set(), 'out': set()}
                for m in n.inbounds:
                    if m not in G:
                        G[m] = {'in': set(), 'out': set()}
                    G[n]['out'].add(m)
                    G[m]['in'].add(n)
                    layers.append(m)

            S = set([input_layers])
            while len(S) > 0:
                n = S.pop()

                graph.append(n)
                for m in n.inbounds:
                    G[n]['out'].remove(m)
                    G[m]['in'].remove(n)
                    # if no other incoming edges add to S
                    if len(G[m]['in']) == 0:
                        S.add(m)

            return graph




    def compile(self, optimizer, loss):
        assert self.inputs is not None and self.outputs is not None

        self.forward_graph,self.trainable_variables=self.topological_sort(self.inputs,mode='forward')
        self.backward_graph=self.topological_sort(self.outputs,mode='backward')

        self.loss = get_objective(loss)
        self.optimizer = get_optimizer(optimizer)



    def fit(self, X, Y, batch_size=64, epochs=20, shuffle=True, validation_data=None, validation_ratio=0.1,draw_acc_loss=False, draw_save_path=None):

        if validation_data is None:
            if 0. < validation_ratio < 1.:
                split = int(X.shape[0] * validation_ratio)
                valid_X, valid_Y = X[-split:], Y[-split:]
                train_X, train_Y = X[:-split], Y[:-split]
                validation_data = (valid_X, valid_Y)
            else:
                train_X, train_Y = X, Y
        else:
            valid_X, valid_Y = validation_data
            train_X, train_Y = X, Y

        for epoch in range(epochs):
            mini_batches = get_batches(train_X, train_Y, batch_size, epoch, shuffle)
            batch_nums = len(mini_batches)
            training_size = train_X.shape[0]
            batch_count = 0
            trained_nums=0
            print('\033[0;31m Epoch[%d/%d]' % (epoch + 1, epochs))
            start_time = time.time()
            for xs, ys in mini_batches:
                batch_count += 1
                trained_nums += xs.shape[0]
                # forward
                y_hat = self.predict(xs)

                #backward
                self.calc_gradients(y_hat,ys)

                end_time = time.time()
                gap = end_time - start_time

                self.optimizer.update(self.trainable_variables)

                batch_acc, batch_loss = self.__evaluate(y_hat, ys)

                self.train_loss.append(batch_loss)
                self.train_acc.append(batch_acc)
                if validation_data is not None:
                    valid_acc, valid_loss = self.evaluate(valid_X, valid_Y,batch_size=batch_size)
                    self.valid_loss.append(valid_loss)
                    self.valid_acc.append(valid_acc)

                if draw_acc_loss:
                    if len(self.train_loss) == 2:
                        plt.ion()
                        plt.figure(figsize=(6, 7))
                        plt.title('batch-size=' + str(batch_size) + ',Epochs=' + str(epochs))
                        ax1 = plt.subplot(2, 1, 1)
                        ax2 = plt.subplot(2, 1, 2)
                    if len(self.train_loss) > 1:
                        self.draw_training(ax1, ax2, draw_save_path, epoch)

                trained_process_bar_nums = batch_count * self.process_bar_nums // batch_nums
                process_bar = self.process_bar_trained * trained_process_bar_nums + '>' + self.process_bar_untrain * ( self.process_bar_nums - trained_process_bar_nums - 1)
                if validation_data is not None:
                    print( '\r{:d}/{:d} [{}] -{:.0f}s -{:.0f}ms/batch -batch_loss: {:.4f} -batch_acc: {:.4f} -val_loss: {:.4f} -val_acc: {:.4f}'.format(trained_nums,training_size,process_bar, gap, gap*1000/batch_count,batch_loss, batch_acc, valid_loss, valid_acc), end='')
                else:
                    print('\r{:d}/{:d} [{}] -{:.0f}s -{:.0f}ms/batch -batch_loss: {:.4f} -batch_acc: {:.4f} '.format(trained_nums,training_size,process_bar, gap,gap*1000/batch_count, batch_loss, batch_acc), end='')
            print()




    def predict(self, X, is_training=True):
        self.inputs.input_tensor = X
        for node in self.forward_graph:
            node.forward(is_training=is_training)
        y_hat = self.outputs.output_tensor
        return y_hat




    def calc_gradients(self,y_hat,y_true):
        self.outputs.grads=self.loss.backward(y_hat,y_true)
        for node in self.backward_graph:
            node.backward()





    def __evaluate(self,y_hat,y_true):
        acc = self.loss.calc_acc(y_hat,y_true)
        base_loss = self.loss.calc_loss(y_hat, y_true)

        return acc,base_loss






    def evaluate(self, X, Y, batch_size=None):
        if batch_size is not None:
            assert type(batch_size) is int
            ep = 0
            acc_list = []
            loss_list = []
            data_nums = X.shape[0]
            while True:
                sp = ep
                ep = min(sp + batch_size, data_nums)
                y_hat = self.predict(X[sp:ep], is_training=False)
                acc = self.loss.calc_acc(y_hat,Y[sp:ep])
                acc_list.append(acc)
                base_loss = self.loss.calc_loss(y_hat, Y[sp:ep])
                loss_list.append(base_loss)

                if ep == data_nums:
                    acc = sum(acc_list) / len(acc_list)
                    base_loss = sum(loss_list) / len(loss_list)
                    break
        else:
            y_hat = self.predict(X, is_training=False)
            acc = self.loss.calc_acc(y_hat,Y)
            base_loss = self.loss.calc_loss(y_hat, Y)
            regular_loss = 0
            # for layer in self.layers:
            #     regular_loss+=layer.add_loss
        return acc, base_loss




    def draw_training(self, ax1, ax2, draw_save_path, epoch):
        leg1 = ax1.get_legend()
        ax1.plot(self.train_loss, color='blue', label='train')
        if self.valid_loss:
            ax1.plot(self.valid_loss, color='green', label='validation')
        ax1.set_xlabel('iter')
        ax1.set_ylabel('loss')
        if leg1 is None:
            ax1.legend(loc='best')
        leg2 = ax2.get_legend()
        ax2.plot(self.train_acc, color='red', label='train')
        if self.valid_acc:
            ax2.plot(self.valid_acc, color='yellow', label='validation')
        ax2.set_xlabel('iter')
        ax2.set_ylabel('acc')
        if leg2 is None:
            ax2.legend(loc='best')
        plt.pause(0.1)
        if draw_save_path is not None:
            assert draw_save_path.__class__.__name__ == 'str'
            draw_save_path = os.path.abspath(draw_save_path + '\\Epoch' + str(epoch))
            plt.savefig(draw_save_path, dpi=300)



    def pop(self, index=-1):
        layer = self.layers.pop(index)
        del layer
        print('success delete %s layer' % (layer.__class__.__name__))



    def save(self, save_path):
        with open(save_path + '.pkl', 'wb') as f:
            pickle.dump([self.forward_graph, self.backward_graph, self.optimizer, self.loss], f)



    def load(self, model_path):
        with open(model_path + '.pkl', 'rb') as f:
            f_graph, b_graph, optimizer, loss = pickle.load(f)

        self.forward_graph = f_graph
        self.backward_graph = b_graph
        self.optimizer = optimizer
        self.loss = loss



    def __str__(self):
        bar_nums = 75
        print('*' * bar_nums)

        print('Layer(type)'.ljust(20),'Output Shape'.ljust(20) ,'Param'.ljust(12),'Connected to'.ljust(15))
        print('#' * bar_nums)
        total_params = 0
        for layer in self.forward_graph:

            if layer.name is not None:
                layer_name = '%s (%s)'%(layer.name,layer.__class__.__name__)
            else:
                layer_name = str(layer.__class__.__name__)

            params = layer.params_count()
            total_params += params
            first = True
            if layer.inbounds:

                for prev_layer in layer.inbounds:
                    if prev_layer.name is not None:
                        connected = prev_layer.name
                    else:
                        connected = prev_layer.__class__.__name__
                    if first:
                        print(layer_name.ljust(20),str(layer.output_shape).ljust(20), str(params).ljust(12),connected.ljust(15))
                        first = False
                    else:
                        print(''.ljust(20),''.ljust(20), ''.ljust(12),connected.ljust(15))
            else:
                connected = '\n'
                print(layer_name.ljust(20),str(layer.output_shape).ljust(20), str(params).ljust(12),connected.ljust(15))
            print('-' * bar_nums)

        print('*' * bar_nums)
        trainable_params = 0
        for v in self.trainable_variables:
            trainable_params += v.output_tensor.size
        params_details = 'Total params: %d\n'%(total_params)
        params_details += 'Trainable params: %d\n'%(trainable_params)
        params_details += 'Non-trainable params: %d\n' % (total_params-trainable_params)
        return params_details



