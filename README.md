# Shinnosuke : Deep Learning Framework

<div align=center>
	<img src="https://gss1.bdstatic.com/9vo3dSag_xI4khGkpoWK1HF6hhy/baike/w%3D268%3Bg%3D0/sign=625eaa79a864034f0fcdc50097f81e0c/8c1001e93901213f91ab2a7857e736d12e2e95fd.jpg" width="">
</div>



## Descriptions

Shinnosuke is a high-level neural network's API that almost identity to Keras with slightly differences. It was written by Python only, and dedicated to realize experimentations quickly.

Here are some features of Shinnosuke:

1. Based on **Numpy** (CPU version)  and **native** to Python.
2. Without any other 3rd-party library.
3. **Keras-like API**, several basic AI Examples are provided, easy to get start.
4. Support commonly used models such as: Dense, Conv2D, MaxPooling2D, LSTM, SimpleRNN, etc.
5. Several basic AI Examples are provided.
6. **Sequential** model (for most  sequence network combinations ) and **Functional** model (for resnet, etc) are implemented.
7. Training is conducted on forward graph and backward graph, meanwhile **autograd** is supported .

Shinnosuke is compatible with: **Python 3.x (3.6 is recommended)**

------



## Getting started

The core networks of Shinnosuke is a model, which provide a way to combine layers. There are two model types: `Sequential` (a linear stack of layers) and `Functional` (build  a graph for layers).

Here is a example of `Sequential` model:

```python
from shinnosuke.models import Sequential

m=Sequential()
```

Using `.add()` to connect layers:

```python
from shinnosuke.models import Dense

m.add(Dense(n_out=500, activation='relu', n_in=784))  #must be specify n_in if current layer is the first layer of model
m.add(Dense(n_out=10, activation='softmax'))  #no need to specify n_in as shinnosuke will automatic calculate the input and output shape
```

Here are some differences with Keras, n_out and n_in are named units and input_dim in Keras respectively.

Once you have constructed your model, you should configure it with `.compile()` before training:

```python
m.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
```

If you apply softmax to multi-classify task and your labels are one-hot encoded vectors/matrix, you shall specify loss as  sparse_categorical_crossentropy, otherwise use categorical_crossentropy. (While in Keras categorical_crossentropy supports for one-hot encoded labels). And Shinnosuke model only supports one metrics--**accuracy**, which no need to specify in `compile`.

 You can further configure your optimizer by passing more parameters:

```python
m.compile(loss=shinnosuke.Objectives.SparseCategoricalCrossEntropy, optimizer='sgd', learning_rate=0.01, epsilon=1e-8)
```

Having finished `compile`, you can start training your data in batches:

```python
#trainX and trainy are Numpy arrays --for 2 dimension data, trainX's shape should be (training_nums,input_dim)
m.fit(trainX, trainy, batch_size=128, epochs=5, validation_ratio=0., draw_acc_loss=True)
```

By specify `validation_ratio`=`(0.0,1.0]`, shinnosuke will split validation data from training data according to validation_ratio, otherwise validation_ratio=0. means no validation data. Alternatively you can  feed validation_data manually:

```python
m.fit(trainX, trainy, batch_size=128, epochs=5, validation_data=(validX,validy), draw_acc_loss=True)
```

If `draw_acc_loss`=**True**, a dynamic updating figure will be shown in the training process, like below:

![draw_acc_loss](https://github.com/eLeVeNnN/shinnosuke/blob/master/docs/imgs/draw_acc.png)



Evaluate your model performance by `.evaluate()`:

```python
acc, loss = m.evaluate(testX, testy, batch_size=128)
```

Or obtain predictions on new data:

```python
y_hat = m.predict(x_test)
```



For `Functional` model, first instantiate an `Input` layer:

```python
from shinnosuke.layers.Base import Input

X_input=Input(shape=(None,1,28,28))   #(batch_size,channels,height,width)
```

You need to specify the input shape, notice that for Convolutional networks,data's channels must be in the `axis 1` instead of `-1`, and you should state batch_size as None which is unnecessary in Keras.

Then Combine your layers by functional API:

```python
from shinnosuke.models import Model
from shinnosuke.layers.Convolution import Conv2D,MaxPooling2D
from shinnosuke.layers.Activation import Activation
from shinnosuke.layers.Normalization import BatchNormalization
from shinnosuke.layers.FC import Flatten,Dense

X=Conv2D(8,(2,2),padding='VALID',initializer='normal',activation='relu')(X_input)
X=MaxPooling2D((2,2))(X)
X=Flatten()(X)
X=Dense(10,initializer='normal',activation='softmax')(X)
model=Model(inputs=X_input,outputs=X)  
model.compile(optimizer='sgd',loss='sparse_categorical_cross_entropy')
model.fit(trainX,trainy,batch_size=256,epochs=80,validation_ratio=0.)
```

Pass inputs and outputs layer to `Model()`, and then compile and fit model like `Sequential`model.



Building an image classification model, a question answering system or any other model is just as convenient and fast~

In the [Examples folder](https://github.com/eLeVeNnN/shinnosuke/Examples) of this repository, you can find more advanced models.(waiting to implement..)

------



## Installation

Before installing Shinnosuke, please install the following **dependencies**:

- Numpy=1.15.0 (recommend)

- matplotlib=3.0.3 (recommend)

Then you can install Shinnosuke by using pip:

`$ pip install shinnosuke`

**Installation from Github source will be supported in the future.**

------



## Supports

### Two basic class:

#### - Layer:

- Dense
- Conv2D
- MaxPooling2D
- MeanPooling2D
- Activation
- Input
- Dropout
- BatchNormalization
- TimeDistributed
- SimpleRNN
- LSTM
- GRU (waiting for implemented)
- ZeroPadding2D
- **Operations( includes Add, Minus, Multiply, Matmul, and so on basic operations for Layer and Node)**

Operations for layers are conducted to construt a graph.(waiting to implement)



#### - Node:

- Variable
- Constant

While Node Operations have both dynamic graph and static graph features

```python
x=Variable(3)
y=Variable(5)
z=x+y
print(z.get_value())
```

You suppose get a value 8,at same time shinnosuke construct a graph as below(waiting to implement):



### Optimizers

- StochasticGradientDescent
- Momentum
- RMSprop
- AdaGrad
- AdaDelta
- Adam

Waiting for implemented more

### Objectives

- MeanSquaredError
- MeanAbsoluteError
- BinaryCrossEntropy
- SparseCategoricalCrossEntropy
- CategoricalCrossEntropy 

### Activations

- Relu
- Linear
- Sigmoid
- Tanh
- Softmax

### Initializations

- Zeros
- Ones
- Uniform
- LecunUniform
- GlorotUniform
- HeUniform
- Normal
- LecunNormal
- GlorotNormal
- HeNormal
- Orthogonal

### Regularizes

waiting for implement.

### Utils

- get_batches (generate mini-batch)
- to_categorical (convert inputs to one-hot vector/matrix)
- concatenate (concatenate Nodes that have the same shape in specify axis)
- pad_sequences (pad sequences to the same length)
