from keras.layers import Input, Dense
import tensorflow as tf
def create_tensors(input):
    x = Input(shape=(input,))
    return x

def isTensor(input):
    if isinstance(input, tf.Tensor):
        return True
    return False

def isDense(input):
    if isinstance(input, type(Dense)):
        return True
    return False

def input_neuron(inputs, outputs=[1,1]):
    input_tensors = []
    for input in inputs:
        if(isTensor(input)):
            input_tensors.append(input)
        else:
            input_tensors.append(create_tensors(input))

    x = Dense(outputs[0])(input_tensors[0])
    x1 = Dense(outputs[1])(input_tensors[1])
    return x, x1

def output_neuron(x, output=1):
    x = Dense(output)(x)

    return x
