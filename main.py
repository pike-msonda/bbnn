from bbnn.bbnn import output_neuron, input_neuron
import numpy as np

if __name__ == "__main__":
    out, out1 = input_neuron(inputs=[1,1])
    out2, out3 = input_neuron(inputs=[out,out1])

    print(out2)
    print(out3)

    