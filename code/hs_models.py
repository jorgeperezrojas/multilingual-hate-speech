import keras
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Masking, BatchNormalization

def LSTM_model(
        max_len = 40,
        vector_size = 300,
        num_layers = 2,
        dims = [128,64],
        bidirectional = True,
        dropout = [0.5,0.5],
        dense_layers = 1,
        dims_dense = [32],
        dropout_dense = [0.5],
        activations = ['relu'],
        use_batch_norm = True
    ):

    assert num_layers == len(dims)
    assert num_layers == len(dropout)
    assert dense_layers == len(dims_dense)
    assert dense_layers == len(dropout_dense)
    assert dense_layers == len(activations)

    model = keras.models.Sequential()
    model.add(Masking(mask_value=0, input_shape=(max_len, vector_size),trainable = False))

    for i in range(num_layers-1):
        lstm_layer = LSTM(dims[i], return_sequences=True)
        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)
        model.add(lstm_layer)
        if dropout[i] > 0.0:
            model.add(Dropout(dropout[i]))

    last = num_layers-1
    last_lstm_layer = LSTM(dims[last], return_sequences=False)
    if bidirectional:
        last_lstm_layer = Bidirectional(last_lstm_layer)
    model.add(last_lstm_layer)
    if dropout[last] > 0.0:
        model.add(Dropout(dropout[last]))

    if use_batch_norm:
        model.add(BatchNormalization())
    for i in range(dense_layers):
        model.add(Dense(dims_dense[i], activation=activations[i]))
        if use_batch_norm:
            model.add(BatchNormalization())
        if dropout_dense[i] > 0.0:
            model.add(Dropout(dropout_dense[i]))

    model.add(Dense(1, activation='sigmoid'))
    return model