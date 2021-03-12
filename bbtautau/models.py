from keras.models import Model
from keras.layers import Input, Dense

def keras_model(n_variables):
    x_1 = Input(shape=n_variables)
    hidden_1 = Dense(64, activation='relu', kernel_initializer='normal')(x_1)
    hidden_2 = Dense(32, activation='relu')(hidden_1)
    hidden_3 = Dense(8, activation='relu')(hidden_2)
    output = Dense(1, activation="linear")(hidden_3)
    return Model(inputs=x_1, outputs=output)
    
