
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dense, SeparableConv2D, Conv1D

from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model


def LHC_DNN (in_shape=(16), n_classes=5, config_params=[]):
    
    x = x_in = Input(in_shape)
    n = config_params[0] 
    
    x = Dense(int(n), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001))(x)
    #x = BatchNormalization(name='bn1')(x)
    x = Activation(activation='relu', name='relu1')(x)
    x = Dense(int(n/2), name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001))(x)
    x = Activation(activation='relu', name='relu2')(x)
    x = Dense(int(n/2), name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001))(x)
    x = Activation(activation='relu', name='relu3')(x)
    x = Dense(int(n_classes), name='output_dense', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001))(x)
    x_out = Activation(activation='softmax', name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='lhc_dnn')

    return model

def LHC_CNN (in_shape=(16), n_classes=5, config_params=[]):
    x = x_in = Input(in_shape)

    x = Conv1D(8, (4), name='conv1', activation='relu')(x)
    x = Conv1D(8, (4), name='conv2', activation='relu')(x)
    x = Conv1D(8, (4), name='conv3', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu', name='fc1')(x)
    x = Dense(16, activation='relu', name='output_dense')(x)
    x_out = Dense(5, activation='softmax', name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='lhc_dnn')

    return model
