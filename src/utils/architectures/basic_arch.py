from tensorflow.keras.layers import Dense, Flatten, Convolution2D


def simple_dense(net_input, n_classes,
                 n_neurons=(8, 4, 4, 4)):
    """
        simple dense regression arch
    """
    output = net_input

    for n_neuron in n_neurons:

        output = Dense(n_neuron,
                     kernel_initializer='normal',
                     activation='relu')(output)

    output = Flatten()(output)
    output = Dense(n_classes,
                   kernel_initializer='normal',
                   activation='linear')(output)
    return output


def simple_cnn(net_input, n_classes,
                 filters=(8, 4, 4, 4),
               kernels=(7, 7, 2, 2)):
    """
        simple Conv regression arch
    """
    output = net_input

    for filters_, kernel_size_ in zip(filters, kernels):

        output = Convolution2D(filters=filters_,
                     kernel_size=(kernel_size_, kernel_size_),
                               padding='same',
                     activation='relu')(output)

    output = Flatten()(output)
    output = Dense(n_classes,
                   kernel_initializer='normal',
                   activation='linear')(output)
    return output