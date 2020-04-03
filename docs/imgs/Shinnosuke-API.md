# Shinnosuke-API

<div align=center>
  <img src="https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=1784511497,119911411&fm=26&gp=0.jpg" width="">
</div>

## Layer

+ *Dense(n_out: int, n_in: int = None, initializer='glorot_uniform', activation='linear', kernel_regularizer=None, **kwargs)*

  + n_out: out feature numbers.

  + n_in: in feature numbers.

  + initializer: kernel and bias initialize method. see details in <a href='#Initializers'>Initializers</a>

  + activation: activation function. see details in <a href='#Activations'>Activations</a>

  + kernel_regularizer: not implemented.

    

+ *Flatten(out_dim: int = 2, **kwargs)*

  + out_dim: after flatten, the output data's dimension. for example, input data's shape is (N, C, H, W), out_dim = 2 will convert output data's shape to (N, $C \times H \times W$) and out_dim = 3 will convert output data's shape to (N, C, $H \times W$).

    

+ *Conv2D(filter_nums: int, filter_size: Tuple, input_shape: Tuple = None, stride: int = 1, padding: str = 'VALID', activation = 'linear',initializer = 'Normal', **kwargs)*

  + filter_nums: filter's numbers.

  + filter_size: filter's size. for example, (3, 3) or 3.

  + input_shape: must specify if this is the first layer of network.

  + stride: convolution stride.

  + padding: 'SAME' or 'VALID', 'VALID' means no padding, 'SAME' means pad input to get the same output size as input.

  + activation: activation function. see details in <a href='#Activations'>Activations</a>

  + initializer: kernel and bias initialize method. see details in <a href='#Initializers'>Initializers</a>

    

+ *ZeroPadding2D(pad_size: Tuple, **kwargs)*

  + pad_size: for example, (1, 1), which means pad input(N, C, H, W) to (N, C, H+2, W+2).

  

+ *MaxPooling2D(pool_size: Tuple, stride: int = None, **kwargs)*

  + pool_size: pooling kernel size, for example (2, 2) means apply max pooling in every 2 x 2 area.
  + stride: pooling stride.

  

+ *MeanPooling2D(pool_size: Tuple, stride: int = None, **kwargs)*

  + pool_size: pooling kernel size, for example (2, 2) means apply mean pooling in every 2 x 2 area.
  + stride: pooling stride.

  

+ *Activation(act_name='relu')*

  + act_name: activation function name, support ReLU, Sigmoid, etc. see details in <a href='#Activations'>Activations</a>

  

+ *Input(shape: Tuple,  value: ndarray = None, **kwargs)*

  + shape: input data's shape, for example, (None, C, H, W) or (None, features).
  + value: this layer's input and output tensor's value.

  

+ *Reshape*(output_shape: Tuple, **kwargs)

  + output_shape: shape after reshape operations.

  

+ *Dropout(keep_prob)*

  + keep_prob:  probability of keeping a unit active.

  

+ *Batch Normalization(epsilon=1e-6, momentum=0.9, axis=1, gamma_initializer='ones', beta_initializer='zeros', moving_mean_initializer='zeros', moving_variance_initializer='ones')*
  $$
  u_B = \frac{1}{m} \sum \limits_{i=1}^m x_i  \quad \quad mini-batch \quad mean
  \\
  \sigma_B = \frac{1}{m} \sum \limits_{i=1}^m (x_i - u_B)^2  \quad \quad mini-batch \quad variance
  \\
  \hat x_i = \frac{x_i - u_B}{\sqrt{\sigma_B^2 + \epsilon}}   \quad \quad normalize
  \\
  y_i = \gamma \hat x_i + \beta  \quad \quad scale \quad and \quad shift
  $$

  + epsilon:  $\epsilon$ value.
  + momentum: at training time, we use moving averages to update $u_B$ and $\sigma_B \rightarrow$ $moving\_u = momentum * moving\_u + (1 - momentum) * u_B; moving\_\sigma = momentum * moving\_\sigma + (1 - momentum) * \sigma_B$ 
  + axis: use normalization on which axis, for Dense Layer, it should be 1 or -1, for Convolution Layer, it should be 1.
  + gamma_initializer: initialize $\gamma$ method. see details in <a href='#Initializers'>Initializers</a>
  + beta_initializer: initialize $\beta$ method. see details in <a href='#Initializers'>Initializers</a>
  + moving_mean_initializer: initialize $moving\_u$ method. see details in <a href='#Initializers'>Initializers</a>
  + moving_variance_initializer: initialize $moving\_\sigma$ method. see details in <a href='#Initializers'>Initializers</a>

  

+ *Layer Normalization(epsilon=1e-10, gamma_initializer='ones', beta_initializer='zeros')*
  $$
  u = \frac{1}{CHW} \sum \limits_{i=1}^C \sum \limits_{j=1}^H \sum \limits_{k=1}^W x_{ijk}  \quad \quad sample \quad mean
  \\
  \sigma = \frac{1}{CHW} \sum \limits_{i=1}^C \sum \limits_{j=1}^H \sum \limits_{k=1}^W (x_{ijk} - u)^2  \quad \quad sample \quad variance
  \\
  \hat x = \frac{x - u}{\sqrt{\sigma^2 + \epsilon}}   \quad \quad normalize
  \\
  y = \gamma \hat x + \beta  \quad \quad scale \quad and \quad shift
  $$
  

  + epsilon:  $\epsilon$ value.
  + gamma_initializer: initialize $\gamma$ method. see details in <a href='#Initializers'>Initializers</a>
  + beta_initializer: initialize $\beta$ method. see details in <a href='#Initializers'>Initializers</a>

  

+ *Group Normalization(epsilon=1e-5, G=16,gamma_initializer='ones', beta_initializer='zeros')*

  split channel into G groups, for each group, applying layer normalization separately.
  $$
  \\
  u = \frac{1}{CHW} \sum \limits_{i=1}^C \sum \limits_{j=1}^H \sum \limits_{k=1}^W x_{ijk}  \quad \quad sample \quad mean
  \\
  \sigma = \frac{1}{CHW} \sum \limits_{i=1}^C \sum \limits_{j=1}^H \sum \limits_{k=1}^W (x_{ijk} - u)^2  \quad \quad sample \quad variance
  \\
  \hat x = \frac{x - u}{\sqrt{\sigma^2 + \epsilon}}   \quad \quad normalize
  \\
  y = \gamma \hat x + \beta  \quad \quad scale \quad and \quad shift
  $$
  

  + epsilon:  $\epsilon$ value.
  + G: group numbers.
  + gamma_initializer: initialize $\gamma$ method. see details in <a href='#Initializers'>Initializers</a>
  + beta_initializer: initialize $\beta$ method. see details in <a href='#Initializers'>Initializers</a>

  

+ *Embedding(input_dim, output_dim,embeddings_initializer='uniform', mask_zero=False, input_length=None, **kwargs)*

  + input_dim: to embedded dimension, for example, input data shape is (N, T, L), input_dim is L.
  + out_dim: after embedding dimension, for example, out_dim = E, input data (N, T, L) after embedding's shape is (N, T, E).
  + embeddings_initializer: embedding kernel initialize method. see details in <a href='#Initializers'>Initializers</a>
  + mask_zero: use masks.
  + input_length: must specify if this layer is first layer of network.

  

+ *SimpleRNN(units, activation='tanh', initializer='glorotuniform', recurrent_initializer='orthogonal', return_sequences=False, return_state=False, stateful=False, **kwargs)*

  at every timesteps
  $$
  z^t = W_{aa}\cdot a^{t-1} + W_{xa}\cdot x^t +b_a
  \\
  a^t = activation(z^t)
  $$
  

  + units: rnn hidden unit numbers, for example, units = a, input data (N, T, L) after rnn will output (N, T, a).
  + activation: activation method. see details in <a href='#Activations'>Activations</a>
  + initializer: $W_{xa}$ initialize method. see details in <a href='#Initializers'>Initializers</a>
  + recurrent_initializer: $W_{aa}$ initialize method. see details in <a href='#Initializers'>Initializers</a>
  + return_sequences: if True, return all timesteps a $\rightarrow$ $[a^1, a^2,..., a^t]$; if False, return the last timesteps $a^t$.
  + return_state: if True, return return_sequences' result and all timesteps a.
  + stateful: if True, use last time $a^t$ to initialize this time $a^1$; if False, use 0 to initialize this time $a^1$.

  

+ *LSTM(units, activation='tanh', recurrent_activation='sigmoid', initializer='glorotuniform', recurrent_initializer='orthogonal', unit_forget_bias=True, return_sequences=False, return_state=False, stateful=False, **kwargs)*

  at every timesteps
  $$
  i^t = recurrent\_activation(W_i[a^{t-1}, x^t] + b_i)
  \\
  f^t = recurrent\_activation(W_f[a^{t-1}, x^t] + b_f)
  \\
  \tilde c^t = activation(W_c[a^{t-1}, x^t] + b_c)
  \\
  c^t = f^t \cdot c^{t-1} + i^t \cdot \tilde c^t
  \\
  o^t = recurrent\_activation(W_o[a^{t-1}, x^t] + b_o)
  \\
  a^t = o^t \cdot tanh(c^t)
  $$
  

  + units: lstm hidden unit numbers.
  + activation: activation method. see details in <a href='#Activations'>Activations</a>
  + recurrent_activation: activation method. see details in <a href='#Activations'>Activations</a>
  + initializer: $W_{xa}$ initialize method. see details in <a href='#Initializers'>Initializers</a>
  + recurrent_initializer: $W_{aa}$ initialize method. see details in <a href='#Initializers'>Initializers</a>
  + unit_forget_bias: if True, initialize $f^t$ bias $b_f$ as 1, else 0.
  + return_sequences: if True, return all timesteps a $\rightarrow$ $[a^1, a^2,..., a^t]$; if False, return the last timesteps $a^t$.
  + return_state: if True, return return_sequences' result and all timesteps a.
  + stateful: if True, use last time $a^t$ to initialize this time $a^1$; if False, use 0 to initialize this time $a^1$.

  

+ TimeDistributed(layer, **kwargs)

  + layer: to apply time distributed layer.

     example:

  ```python
  from shinnosuke.layers import LSTM
  from shinnosuke.layers import TimeDistributed
  from shinnosuke.layers import Input
  from shinnosuke.layers import Dense
  
  X = Input(shape=(None, 25, 97))  # (N, 25, 97)
  X = LSTM(units=100, return_sequences=True, stateful=True)(X)  # (N, 25, 100)
  X = TimeDistributed(Dense(50))(X)  # (N, 25, 50)
  ```

  

## Node

+ *Variable(initial_value=None, shape=None,name='variable')*

  + initial_value: initialize value of this variable.
  + shape: this variable's shape.
  + name: this variable's name.

  

+ *Constant(output_tensor, name='constant')*

  + output_tensor: this constant value, once initialized, it can't be changed.
  + name: this constant's name.

     ```python
     from shinnosuke.layers import Constant
     
     a = Constant(5)
     print(a)  
     a.output_tensor = 4
     print(a)
     # result in console
     '''
     5
     UserWarning: Can not change Constant value!
     5
     '''
     ```

  

## Optimizers

+ StochasticGradientDescent(lr=0.01, decay=0.0, *args, **kwargs)
  $$
  w = w - lr * dw
  \\
  b = b - lr * db 
  $$
  

  + lr: learning rate.
  + decay: weight decay.

  

+ Momentum(lr=0.01, decay=0.0, rho=0.9, *args, **kwargs)
  $$
  V_{dw} = rho * V_{dw} + (1 - rho) * dw
  \\
  V_{db} = rho * V_{db} + (1 - rho) * db
  \\
  w = w - lr * V_{dw}
  \\
  b = b - lr * V_{db}
  $$
  

  + lr: learning rate.
  + decay: weight decay.
  + rho: moving averages parameter.

  

+ RMSprop(lr=0.001, decay=0.0, rho=0.9, epsilon=1e-7, *args, **kwargs)
  $$
  S_{dw} = rho * S_{dw} + (1 - rho) * d_w^2
  \\
  S_{db} = rho * S_{db} + (1 - rho) * d_b^2
  \\
  w = w - lr * \frac{dw}{\sqrt{S_{dw} + \epsilon}}
  \\
  b = b - lr * \frac{db}{\sqrt{S_{db} + \epsilon}}
  $$
  

  + lr: learning rate.
  + decay: weight decay.
  + rho: moving averages parameter.
  + epsilon: $\epsilon$ value.

  

+ Adam(lr=0.001, decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-7, *args, **kwargs)
  $$
  V_{dw} = rho * V_{dw} + (1 - rho) * dw
  \\
  V_{db} = rho * V_{db} + (1 - rho) * db
  \\
  S_{dw} = rho * S_{dw} + (1 - rho) * d_w^2
  \\
  S_{db} = rho * S_{db} + (1 - rho) * d_b^2
  \\
  V_{dw}^{corrected} = \frac{V_{dw}}{1 - \beta_1^t}
  \\
  V_{db}^{corrected} = \frac{V_{db}}{1 - \beta_1^t}
  \\
  S_{dw}^{corrected} = \frac{S_{dw}}{1 - \beta_2^t}
  \\
  S_{db}^{corrected} = \frac{S_{db}}{1 - \beta_2^t}
  \\
  w = w - lr * \frac{V_{dw}^{corrected}}{S_{dw}^{corrected}}
  \\
  b = b - lr * \frac{V_{db}^{corrected}}{S_{db}^{corrected}}
  $$
  

  + lr: learning rate.

  + decay: weight decay.

  + beta1: $\beta_1$ value.

  + beta2: $\beta_2$ value.

  + epsilon: $\epsilon$ value.

    

## Objectives

+ *MeanSquaredError*
  + loss = $\frac{1}{2}(y - \hat y)^2$

+ *MeanAbsoluteError*
  + loss = $|y - \hat y|$
+ *BinaryCrossEntropy*
  + loss = $-ylog\hat y -(1-y)log(1 - \hat y)$
+ *SparseCategoricalCrossEntropy*
  + loss = $-\sum \limits_{c=1}^C y_clog\hat y_c$
  + $y_c$ should be one-hot vector.
+ *CategoricalCrossEntropy*
  + loss = $-\sum \limits_{c=1}^C y_clog\hat y_c$
  + $y_c$ can not be ont-hot vector.

<h2 id="Activations">Activations</h2>





<h2 id="Initializers">Initializers</h2>





## Utils

