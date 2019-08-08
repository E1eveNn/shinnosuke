# Shinnosuke : Deep learning framework
## Descriptions
1. Based on Numpy(CPU version)

2. Completely realized by Python only
3. Keras-like API
4. Graph are used to construct the system
5. For deep learning studying

## Features
1. Native to Python

2. Keras-like API
3. Easy to get start
4. Commonly used models are provided: Dense, Conv2D, MaxPooling2D, LSTM, SimpleRNN, etc
5. Several basic networks Examples
6. Sequential model and Functional model are implemented
7. Autograd is supported 
8. training is conducted on forward graph and backward graph

## Installation
Using pip:

`$ pip install shinnosuke`

## Supports

### Two model types:
1.**Sequential**

```python
from shinnosuke.models import Sequential
from shinnosuke.layers.FC import Dense

m=Sequential()

m.add(Dense(500,activation='relu',n_in=784))

m.add(Dense(10,activation='softmax'))

m.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',learning_rate=0.1)

m.fit(trainX,trainy,batch_size=512,epochs=1,validation_ratio=0.)

```
2.**Model**
```python
from shinnosuke.models import Sequential
from shinnosuke.layers.FC import Dense

X_input=Input(shape=(None,784))

X=Dense(500,activation='relu')(X_input)

X=Dense(10,activation='softmax')(X)

model=Model(inputs=X_input,outputs=X)

model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',learning_rate=0.1)

model.fit(trainX,trainy,batch_size=512,epochs=1,validation_ratio=0.)
```
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
- Operations( includes Add, Minus, Multiply, Matmul, and so on basic operations for Layer and Node)

Layer Operations are conducted to construt the graph.
for examples:


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
#you suppose get a value 8,at same time shinnosuke construct a graph as below(waiting to implement):



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

- CategoricalCrossEntropy (waiting for implemented)

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
