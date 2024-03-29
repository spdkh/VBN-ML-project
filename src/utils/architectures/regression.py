"""
    Regression architectures
"""
from tensorflow.keras.layers import Dense, Flatten


def simple_dense(net_input, n_classes,
                 n_neurons=(8, 4, 4, 4)):
    """
        simple dense regression arch
    """
    conv = net_input

    for n_neuron in n_neurons:

        conv = Dense(n_neuron,
                     kernel_initializer='normal',
                     activation='relu')(conv)

    conv = Flatten()(conv)
    output = Dense(n_classes,
                   kernel_initializer='normal',
                   activation='linear')(conv)
    return output
