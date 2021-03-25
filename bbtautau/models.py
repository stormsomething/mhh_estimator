from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Reshape
from keras.layers.recurrent import LSTM

# def keras_model(n_variables):
#     x_1 = Input(shape=n_variables)
#     hidden_1 = Dense(256, activation='relu', kernel_initializer='normal')(x_1)
#     hidden_2 = Dense(32, activation='relu')(hidden_1)
#     hidden_3 = Dense(8, activation='relu')(hidden_2)
#     output = Dense(1, activation="linear")(hidden_3)
#     return Model(inputs=x_1, outputs=output)

def keras_model(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Reshape(target_shape=(16, 1,), input_shape=(n_variables,))(x_1)
    hidden_1 = TimeDistributed(Dense(32, activation='relu'))(hidden)
    hidden_2 = LSTM(128)(hidden_1)
    hidden_3 = Dense(32, activation='relu')(hidden_2)
    hidden_4 = Dense(8, activation='relu')(hidden_3)
    output = Dense(1, activation="linear")(hidden_4)
    return Model(inputs=x_1, outputs=output)

