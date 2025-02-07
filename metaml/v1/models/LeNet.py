
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dense, SeparableConv2D

from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model

def LeNet5 (in_shape, n_classes,config_params=[]) :
        
    filters = config_params[0]
    fc_num  = config_params[1]
    cnn_flt_num = [filters, filters*2]
    fc_out_num  = [fc_num]

    x = x_in = Input(in_shape)

    for i in range(2):
        x = Conv2D(filters=cnn_flt_num[i], kernel_size=(5,5), 
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                    padding="same", name=f'conv_{i}')(x) 
        x = Activation('relu',name=f'conv_act_{i}')(x)
        x = MaxPooling2D(pool_size = (2,2),name=f'pool_{i}' )(x)

    x = Flatten()(x)
    for i,n in enumerate(fc_out_num):
        x = Dense(n,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),name=f'dense_{i}')(x)
        x = Activation('relu',name=f'dense_act_{i}')(x)

    x = Dense(int(n_classes),name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model_lenet5')

    return model

def LeNet5_v4096 (in_shape, n_classes,config_params=[]) :
        
    filters = config_params[0]
    fc_num  = config_params[1]
    cnn_flt_num = [filters, filters]
    fc_out_num  = [fc_num]

    x = x_in = Input(in_shape)

    for i in range(2):
        x = Conv2D(filters=cnn_flt_num[i], kernel_size=(5,5), 
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                    padding="same", name=f'conv_{i}')(x) 
        x = Activation('relu',name=f'conv_act_{i}')(x)
        x = MaxPooling2D(pool_size = (2,2),name=f'pool_{i}' )(x)

    x = Flatten()(x)
    for i,n in enumerate(fc_out_num):
        x = Dense(n,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),name=f'dense_{i}')(x)
        x = Activation('relu',name=f'dense_act_{i}')(x)

    x = Dense(int(n_classes),name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model_lenet5')

    return model
