from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Reshape, Masking
from keras.layers.recurrent import LSTM
from bbtautau.SumLayer import SumLayer

# def keras_model(n_variables):
#     x_1 = Input(shape=n_variables)
#     hidden_1 = Dense(256, activation='relu', kernel_initializer='normal')(x_1)
#     hidden_2 = Dense(32, activation='relu')(hidden_1)
#     hidden_3 = Dense(8, activation='relu')(hidden_2)
#     output = Dense(1, activation="linear")(hidden_3)
#     return Model(inputs=x_1, outputs=output)

#original RNN -- 4427
def keras_model_1(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Reshape(target_shape=(18, 1,), input_shape=(n_variables,))(x_1)
    hidden_1 = TimeDistributed(Dense(32, activation='relu'))(hidden)
    hidden_2 = LSTM(128, activation='relu')(hidden_1) #used to be 128
    hidden_3 = Dense(32, activation='relu')(hidden_2)
    hidden_4 = Dense(8, activation='relu')(hidden_3)
    output = Dense(1, activation="linear")(hidden_4) # must be kept 'linear'
    return Model(inputs=x_1, outputs=output)

# Improvement! This one seems like a winner...
def keras_model_2(n_variables):
    x_1 = Input(shape=n_variables) # (None, 18)
    mask = Masking(mask_value=0.0)(x_1) # (None, 18)
    hidden = Reshape(target_shape=(18, 1,), input_shape=(n_variables,))(mask) # (None, 18, 1)
    hidden_1 = TimeDistributed(Dense(40, activation='relu'))(hidden) # (None, 18, 40). Tuned. (Originally 32).
    hidden_2 = LSTM(48)(hidden_1) # (None, 48). Tuned. (Originally 64).
    hidden_3 = Reshape(target_shape=(48,1,), input_shape=(48,))(hidden_2) # (None, 48, 1)
    hidden_4 = TimeDistributed(Dense(8, activation='relu'))(hidden_3) # (None, 48, 8)
    hidden_5 = Reshape(target_shape=(384,), input_shape=(48,8,))(hidden_4) # (None, 384)
    hidden_6 = Dense(32, activation='relu')(hidden_5) # (None, 32)
    hidden_7 = Dense(8, activation='relu')(hidden_6) # (None, 8)
    output = Dense(1, activation='linear')(hidden_7) # (None, 1)
    return Model(inputs=x_1, outputs=output)

#val_loss = 4469
def keras_model_3(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Reshape(target_shape=(18, 1,), input_shape=(n_variables,))(x_1)
    hidden_1 = TimeDistributed(Dense(32, activation='relu'))(hidden)
    hidden_2 = LSTM(64)(hidden_1)
    hidden_3 = Reshape(target_shape=(64,1,), input_shape=(64,))(hidden_2)
    hidden_4 = TimeDistributed(Dense(8, activation='relu'))(hidden_3)
    hidden_5 = Reshape(target_shape=(512,), input_shape=(64,8,))(hidden_4)
    hidden_6 = Dense(32, activation='relu')(hidden_5)
    hidden_7 = Dense(32, activation='relu')(hidden_6)
    output = Dense(1, activation="linear")(hidden_7)
    return Model(inputs=x_1, outputs=output)

# trying to emulate tau ID algorithm -- no good I think!
def keras_model_4(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Reshape(target_shape=(18,1,), input_shape=(n_variables,))(x_1)
    hidden_1 = TimeDistributed(Dense(32, activation='relu'))(hidden)
    hidden_2 = TimeDistributed(Dense(32, activation='relu'))(hidden_1)
    hidden_3 = LSTM(128)(hidden_2) # Sum_Layer()
    hidden_4 = Dense(32, activation='relu')(hidden_3)
    hidden_5 = Dense(32, activation='relu')(hidden_4)
    hidden_6 = Dense(16, activation='relu')(hidden_5)
    hidden_7 = Dense(8, activation='relu')(hidden_6)
    output = Dense(1, activation='linear')(hidden_7)
    return Model(inputs=x_1, outputs=output)
