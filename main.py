from bbnn.bbnn import output_neuron, input_neuron, hidden_neuron, add_dense_layer, make_model
import numpy as np

from keras.utils import plot_model

if __name__ == "__main__":

    out, out1, in0, in1 = input_neuron(inputs=[1,1])  #create entry points in the network

    out2, out3, in2, in3 = input_neuron(inputs=[1,1]) #create entry points in the network

    hid1 =  hidden_neuron(inputs=[out, out1]) # hidden layer

    hid2 =  hidden_neuron(inputs=[out2, out3]) #hidden layer

    output =  add_dense_layer([hid1, hid2]) #adding hidden layers into one.

    x = output_neuron(output) #output layer

    model = make_model(input=[in0,in1, in2, in3], output=x) #creating model

    model.summary() 

    # plot_model(model, to_file='sample.png')  #ignore.


