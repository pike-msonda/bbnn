from keras.layers import Input, Dense
from keras.layers import Add
from keras.models import Model
import tensorflow as tf

"""
    Create tensor.
        Check Tensorflow documentation under Tensors
"""
def create_tensors(input):
    x = Input(shape=(input,))
    return x

"""
    Check if the input is a tensor object.
"""
def isTensor(input):
    if isinstance(input, tf.Tensor):
        return True
    return False

def isDense(input):
    if isinstance(input, type(Dense)):
        return True
    return False

"""
    Input Neuron. If input is not tensor. Create a tensor. 
    output tensor
"""
def input_neuron(inputs, output=1):
    input_tensors = []
    for input in inputs:
        if(isTensor(input)):
            input_tensors.append(input)
        else:
            input_tensors.append(create_tensors(input))

    x = Dense(output, activation='relu')(input_tensors[0])
    x1 = Dense(output, activation='relu')(input_tensors[1])
    return x, x1, input_tensors[0], input_tensors[1]


def hidden_neuron(inputs, outputs=[1,1]):
    hidden_tensors=[]
    for input, output in zip(inputs,outputs):
        hidden_tensors.append(Dense(output)(input))
    
    return hidden_tensors


def add_dense_layer(dense):
    tensors_to_add = []
    for d in dense:
        for t in d:
            tensors_to_add.append(t)
        
    return Add()(tensors_to_add)

"""
    output layer function. 
"""
def output_neuron(x, output=1):
    x = Dense(output, activation="softmax")(x)

    return x

def make_model(input, output):
    model = Model(inputs=input, outputs=output)

    return model