from keras.layers import Input, Dense


def input_neuron(input=2, output=2):
    x =  Input(shape=(input,)) #create tensor 
    x = Dense(output)(x)

    return x
    
def hidden_neuron(x, output=2):
    x = Dense(output)(x)

    return x

def output_neuron(x, output=2):
    x = Dense(output)(x)

    return x
