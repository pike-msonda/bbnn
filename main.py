from bbnn.bbnn import output_neuron, create_tensors, input_neuron
import numpy as np

if __name__ == "__main__":
    out, out1 = input_neuron(inputs=[1,1])
    # import pdb; pdb.set_trace()
    out2, out3 = input_neuron(inputs=[1,out1])

    print(out2)
    print(out3)

    