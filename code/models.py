import keras
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Masking

def LSTM_model(
        max_len = 40,
        num_layers = 3,
        dims = [128,128,32],
        bidirectional = True,
        dropout = [0.3,0.3,0.3],
        dense_layers = 2,
        dims_dense = [32,32],
        dropout_dense = [0.4,0.4],
        activations = ['relu', 'relu']
    ):

    assert num_layers == len(dims)
    assert num_layers == len(dropout)

    model = keras.Sequential()
    model.add(Masking(mask_value=0, input_shape=(max_len, vector_size),trainable = False))

    for i in range(num_layers-1):
        lstm_layer = LSTM(dims[i], return_sequences=True)
        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)
        model.add(lstm_layer)
        if dropout[i] > 0.0:
            model.add(Dropout(dropout[i]))

    last = num_layers-1
    last_layer = LSTM(dims[last], return_sequences=False)
    if bidirectional:
        last_layer = Bidirectional(last_layer)
    if dropout[last] > 0.0:
        model.add(Dropout(dropout[last]))

    for i in range(dense_layers):
        model.add(Dense(dims_dense[i], activation=activations[i]))
        if dropout_dense[i] > 0.0:
            model.add(Dropout(dropout_dense[i]))

    model.add(Dense(1, activation='sigmoid'))
    return model