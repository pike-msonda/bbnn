from bbnn.bbnn import output_neuron, input_neuron, hidden_neuron, add_dense_layer, make_model, show_weights
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import numpy as np

from keras.utils import plot_model


"""
This is a model that create a simple neuron block with two inputs and two outputs. 
"""
if __name__ == "__main__":

    out, out1, in0, in1 = input_neuron(inputs=[1,1])  #create entry points in the network and 
    import pdb; pdb.set_trace()
    model = make_model(input=[in0,in1], output=[out,out1]) #creating model

    model.summary() 

    plot_model(model, to_file='sample.png')  #ignore.
