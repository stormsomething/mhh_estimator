from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Reshape, Masking
from keras.layers.recurrent import LSTM
from bbtautau.SumLayer import SumLayer

def keras_model_main(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Dense(1024, activation='relu')(x_1)
    hidden_1 = Dense(512, activation='relu')(hidden)
    hidden_2 = Dense(256, activation='relu')(hidden_1)
    hidden_3 = Dense(128, activation='relu')(hidden_2)
    hidden_4 = Dense(64, activation='relu')(hidden_3)
    hidden_5 = Dense(32, activation='relu')(hidden_4)
    hidden_6 = Dense(16, activation='relu')(hidden_5)
    hidden_7 = Dense(8, activation='relu')(hidden_6)
    output = Dense(1, activation='linear')(hidden_7)
    return Model(inputs=x_1, outputs=output)
    
def keras_model_triangle_512_64(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Dense(512, activation='relu')(x_1)
    hidden_1 = Dense(448, activation='relu')(hidden)
    hidden_2 = Dense(384, activation='relu')(hidden_1)
    hidden_3 = Dense(320, activation='relu')(hidden_2)
    hidden_4 = Dense(256, activation='relu')(hidden_3)
    hidden_5 = Dense(192, activation='relu')(hidden_4)
    hidden_6 = Dense(128, activation='relu')(hidden_5)
    hidden_7 = Dense(64, activation='relu')(hidden_6)
    output = Dense(1, activation='linear')(hidden_7)
    return Model(inputs=x_1, outputs=output)
    
def keras_model_funnel_512(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Dense(512, activation='relu')(x_1)
    hidden_1 = Dense(256, activation='relu')(hidden)
    hidden_2 = Dense(128, activation='relu')(hidden_1)
    hidden_3 = Dense(64, activation='relu')(hidden_2)
    hidden_4 = Dense(32, activation='relu')(hidden_3)
    hidden_5 = Dense(16, activation='relu')(hidden_4)
    hidden_6 = Dense(8, activation='relu')(hidden_5)
    output = Dense(1, activation='linear')(hidden_6)
    return Model(inputs=x_1, outputs=output)
    
def keras_model_64_5(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Dense(64, activation='relu')(x_1)
    hidden_1 = Dense(64, activation='relu')(hidden)
    hidden_2 = Dense(64, activation='relu')(hidden_1)
    hidden_3 = Dense(64, activation='relu')(hidden_2)
    hidden_4 = Dense(64, activation='relu')(hidden_3)
    output = Dense(1, activation='linear')(hidden_4)
    return Model(inputs=x_1, outputs=output)
    
def keras_model_32_10_juggle(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Dense(32, activation='relu')(x_1)
    hidden_1 = Dense(32, activation='selu')(hidden)
    hidden_2 = Dense(32, activation='relu')(hidden_1)
    hidden_3 = Dense(32, activation='selu')(hidden_2)
    hidden_4 = Dense(32, activation='relu')(hidden_3)
    hidden_5 = Dense(32, activation='selu')(hidden_4)
    hidden_6 = Dense(32, activation='relu')(hidden_5)
    hidden_7 = Dense(32, activation='selu')(hidden_6)
    hidden_8 = Dense(32, activation='relu')(hidden_7)
    hidden_9 = Dense(32, activation='selu')(hidden_8)
    output = Dense(1, activation='linear')(hidden_9)
    return Model(inputs=x_1, outputs=output)
    
def keras_model_32_5(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Dense(32, activation='relu')(x_1)
    hidden_1 = Dense(32, activation='relu')(hidden)
    hidden_2 = Dense(32, activation='relu')(hidden_1)
    hidden_3 = Dense(32, activation='relu')(hidden_2)
    hidden_4 = Dense(32, activation='relu')(hidden_3)
    output = Dense(1, activation='linear')(hidden_4)
    return Model(inputs=x_1, outputs=output)

    
def keras_model_funnel_128(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Dense(128, activation='relu')(x_1)
    hidden_1 = Dense(64, activation='relu')(hidden)
    hidden_2 = Dense(32, activation='relu')(hidden_1)
    hidden_3 = Dense(16, activation='relu')(hidden_2)
    hidden_4 = Dense(8, activation='relu')(hidden_3)
    output = Dense(1, activation='linear')(hidden_4)
    return Model(inputs=x_1, outputs=output)
    
def keras_model_funnel_64(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Dense(64, activation='relu')(x_1)
    hidden_1 = Dense(32, activation='relu')(hidden)
    hidden_2 = Dense(16, activation='relu')(hidden_1)
    hidden_3 = Dense(8, activation='relu')(hidden_2)
    output = Dense(1, activation='linear')(hidden_3)
    return Model(inputs=x_1, outputs=output)

def keras_model_main_old(n_variables):
    x_1 = Input(shape=n_variables)
    mask = Masking(mask_value=0.0)(x_1)
    hidden = Reshape(target_shape=(17, 1,), input_shape=(n_variables,))(mask)
    hidden_1 = TimeDistributed(Dense(40, activation='relu'))(hidden)
    hidden_2 = LSTM(48, activation='relu')(hidden_1)
    hidden_3 = Reshape(target_shape=(48,1,), input_shape=(48,))(hidden_2)
    hidden_4 = TimeDistributed(Dense(8, activation='relu'))(hidden_3)
    hidden_5 = Reshape(target_shape=(384,), input_shape=(48,8,))(hidden_4)
    hidden_6 = Dense(32, activation='relu')(hidden_5)
    hidden_7 = Dense(8, activation='relu')(hidden_6)
    output = Dense(1, activation='linear')(hidden_7)
    return Model(inputs=x_1, outputs=output)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Below are some of other keras models I tried out!

def keras_model_1_1(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Reshape(target_shape=(17, 1,), input_shape=(n_variables,))(x_1)
    hidden_1 = TimeDistributed(Dense(32, activation='relu'))(hidden)
    hidden_2 = LSTM(128, activation='relu')(hidden_1)
    hidden_3 = Dense(32, activation='relu')(hidden_2)
    hidden_4 = Dense(8, activation='relu')(hidden_3)
    output = Dense(1, activation="linear")(hidden_4)
    return Model(inputs=x_1, outputs=output)

def keras_model_1_2(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Reshape(target_shape=(17, 1,), input_shape=(n_variables,))(x_1)
    hidden_1 = TimeDistributed(Dense(32, activation='relu'))(hidden)
    hidden_2 = SumLayer()(hidden_1)
    hidden_3 = Dense(32, activation='relu')(hidden_2)
    hidden_4 = Dense(8, activation='relu')(hidden_3)
    output = Dense(1, activation="linear")(hidden_4)
    return Model(inputs=x_1, outputs=output)

def keras_model_2_1(n_variables):
    x_1 = Input(shape=n_variables)
    mask = Masking(mask_value=0.0)(x_1)
    hidden = Reshape(target_shape=(17, 1,), input_shape=(n_variables,))(mask)
    hidden_1 = TimeDistributed(Dense(64, activation='relu'))(hidden)
    hidden_2 = SumLayer()(hidden_1)
    hidden_3 = Reshape(target_shape=(64,1,), input_shape=(64,))(hidden_2)
    hidden_4 = TimeDistributed(Dense(8, activation='relu'))(hidden_3)
    hidden_5 = Reshape(target_shape=(512,), input_shape=(64,8,))(hidden_4)
    hidden_6 = Dense(32, activation='relu')(hidden_5)
    hidden_7 = Dense(8, activation='relu')(hidden_6)
    output = Dense(1, activation='linear')(hidden_7)
    return Model(inputs=x_1, outputs=output)

def keras_model_2_2(n_variables):
    x_1 = Input(shape=n_variables)
    mask = Masking(mask_value=0.0)(x_1)
    hidden = Reshape(target_shape=(17, 1,), input_shape=(n_variables,))(mask)
    hidden_1 = TimeDistributed(Dense(40, activation='relu'))(hidden)
    hidden_2 = LSTM(48)(hidden_1)
    hidden_3 = Reshape(target_shape=(48,1,), input_shape=(48,))(hidden_2)
    hidden_4 = TimeDistributed(Dense(8, activation='relu'))(hidden_3)
    hidden_5 = Reshape(target_shape=(384,), input_shape=(48,8,))(hidden_4)
    hidden_6 = Dense(64, activation='relu')(hidden_5)
    hidden_7 = Dense(32, activation='relu')(hidden_6)
    hidden_8 = Dense(8, activation='relu')(hidden_7)
    output = Dense(1, activation='linear')(hidden_8)
    return Model(inputs=x_1, outputs=output)

def keras_model_2_3(n_variables):
    x_1 = Input(shape=n_variables)
    mask = Masking(mask_value=0.0)(x_1)
    hidden = Reshape(target_shape=(17, 1,), input_shape=(n_variables,))(mask)
    hidden_1 = TimeDistributed(Dense(40, activation='selu'))(hidden)
    hidden_2 = LSTM(48, activation='selu')(hidden_1)
    hidden_3 = Reshape(target_shape=(48,1,), input_shape=(48,))(hidden_2)
    hidden_4 = TimeDistributed(Dense(8, activation='selu'))(hidden_3)
    hidden_5 = Reshape(target_shape=(384,), input_shape=(48,8,))(hidden_4)
    hidden_6 = Dense(32, activation='selu')(hidden_5)
    hidden_7 = Dense(8, activation='selu')(hidden_6)
    output = Dense(1, activation='linear')(hidden_7)
    return Model(inputs=x_1, outputs=output)

def keras_model_3(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Reshape(target_shape=(17, 1,), input_shape=(n_variables,))(x_1)
    hidden_1 = TimeDistributed(Dense(32, activation='relu'))(hidden)
    hidden_2 = LSTM(64)(hidden_1)
    hidden_3 = Reshape(target_shape=(64,1,), input_shape=(64,))(hidden_2)
    hidden_4 = TimeDistributed(Dense(8, activation='relu'))(hidden_3)
    hidden_5 = Reshape(target_shape=(512,), input_shape=(64,8,))(hidden_4)
    hidden_6 = Dense(32, activation='relu')(hidden_5)
    hidden_7 = Dense(32, activation='relu')(hidden_6)
    output = Dense(1, activation="linear")(hidden_7)
    return Model(inputs=x_1, outputs=output)

def keras_model_4(n_variables):
    x_1 = Input(shape=n_variables)
    hidden = Reshape(target_shape=(17,1,), input_shape=(n_variables,))(x_1)
    hidden_1 = TimeDistributed(Dense(32, activation='relu'))(hidden)
    hidden_2 = TimeDistributed(Dense(32, activation='relu'))(hidden_1)
    hidden_3 = SumLayer()(hidden_2) # Sum_Layer()
    hidden_4 = Dense(32, activation='relu')(hidden_3)
    hidden_5 = Dense(32, activation='relu')(hidden_4)
    hidden_6 = Dense(16, activation='relu')(hidden_5)
    hidden_7 = Dense(8, activation='relu')(hidden_6)
    output = Dense(1, activation='linear')(hidden_7)
    return Model(inputs=x_1, outputs=output)
