from bbnn.bbnn import output_neuron, input_neuron, hidden_neuron, add_dense_layer, make_model
import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.utils import plot_model

if __name__ == "__main__":
    out, out1, in0, in1 = input_neuron(inputs=[1,1]) 

    out2, out3, in2, in3 = input_neuron(inputs=[1,1])

    hid1 =  hidden_neuron(inputs=[out, out1])

    hid2 =  hidden_neuron(inputs=[out2, out3])

    output =  add_dense_layer([hid1, hid2])

    x = output_neuron(output)

    model = make_model(input=[in0,in1, in2, in3], output=x)

    model.summary()

    plot_model(model, to_file='sample.png')


