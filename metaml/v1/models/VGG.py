
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dense, SeparableConv2D

from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model


def VGG6 (in_shape, n_classes,config_params=[]) :
    # filters_per_conv_layer = [4,4,8]
    # neurons_per_dense_layer = [8,16]

    filters_per_conv_layer = [16,16,24]
    neurons_per_dense_layer = [42,64]

    #x = x_in = Input(input_shape)
    x = x_in = Input(in_shape)
    

    for i,f in enumerate(filters_per_conv_layer):
        x = Conv2D(int(f), kernel_size=(3,3), strides=(1,1), kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=False,
                name='conv_{}'.format(i))(x) 
        x = BatchNormalization(name='bn_conv_{}'.format(i))(x)
        x = Activation('relu',name='conv_act_%i'%i)(x)
        x = MaxPooling2D(pool_size = (2,2),name='pool_{}'.format(i) )(x)
    x = Flatten()(x)

    for i,n in enumerate(neurons_per_dense_layer):
        x = Dense(n,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),name='dense_%i'%i, use_bias=False)(x)
        x = BatchNormalization(name='bn_dense_{}'.format(i))(x)
        x = Activation('relu',name='dense_act_%i'%i)(x)
    x = Dense(int(n_classes),name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model')
    return model


def VGG7_v4096 (in_shape, n_classes, config_params) :
    
    filters = config_params[0]
    fc_num  = config_params[1]
    scale_factor = 1.36
    cnn_flt_num = [filters, filters, int(filters*scale_factor), int(filters*scale_factor)]
    fc_out_num = [fc_num, fc_num*2]

    x = x_in = Input(in_shape)

    for i in range(4):
        x = Conv2D(filters=cnn_flt_num[i], kernel_size=(3,3), 
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                    #name=f'conv_{i}')(x) 
                    padding="same", name=f'conv_{i}')(x) 
        x = BatchNormalization(name=f'bn_conv_{i}')(x)
        x = Activation('relu',name=f'conv_act_{i}')(x)
        x = MaxPooling2D(pool_size = (2,2),name=f'pool_{i}' )(x)

    #for i in range(2,3,2):
    #    x = Conv2D(filters=cnn_flt_num[i], kernel_size=(3,3), 
    #                kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
    #                name=f'conv_{i}')(x) 
    #                #padding="same", name=f'conv_{i}')(x) 
    #    x = BatchNormalization(name=f'bn_conv_{i}')(x)
    #    x = Activation('relu',name=f'conv_act_{i}')(x)
    #    x = Conv2D(filters=cnn_flt_num[i+1], kernel_size=(3,3), 
    #                kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
    #                name=f'conv_{i+1}')(x) 
    #                #padding="same", name=f'conv_{i+1}')(x) 
    #    x = BatchNormalization(name=f'bn_conv_{i+1}')(x)
    #    x = Activation('relu',name=f'conv_act_{i+1}')(x)
    #    x = MaxPooling2D(pool_size = (2,2),name=f'pool_{i+1}' )(x)

    x = Flatten()(x)
    for i,n in enumerate(fc_out_num):
        x = Dense(n,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),name=f'dense_{i}')(x)
        x = Activation('relu',name=f'dense_act_{i}')(x)

    x = Dense(int(n_classes),name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model_vgg7')
    return model 

def VGG7_v8192 (in_shape, n_classes, config_params) :
    
    filters = config_params[0]
    fc_num  = config_params[1]
    cnn_flt_num = [filters, filters, int(filters*1.5), int(filters*1.5)]
    fc_out_num = [fc_num, fc_num*2]

    x = x_in = Input(in_shape)

    for i in range(4):
        x = Conv2D(filters=cnn_flt_num[i], kernel_size=(3,3), 
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                    #name=f'conv_{i}')(x) 
                    padding="same", name=f'conv_{i}')(x) 
        x = BatchNormalization(name=f'bn_conv_{i}')(x)
        x = Activation('relu',name=f'conv_act_{i}')(x)
        x = MaxPooling2D(pool_size = (2,2),name=f'pool_{i}' )(x)

    #for i in range(2,3,2):
    #    x = Conv2D(filters=cnn_flt_num[i], kernel_size=(3,3), 
    #                kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
    #                name=f'conv_{i}')(x) 
    #                #padding="same", name=f'conv_{i}')(x) 
    #    x = BatchNormalization(name=f'bn_conv_{i}')(x)
    #    x = Activation('relu',name=f'conv_act_{i}')(x)
    #    x = Conv2D(filters=cnn_flt_num[i+1], kernel_size=(3,3), 
    #                kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
    #                name=f'conv_{i+1}')(x) 
    #                #padding="same", name=f'conv_{i+1}')(x) 
    #    x = BatchNormalization(name=f'bn_conv_{i+1}')(x)
    #    x = Activation('relu',name=f'conv_act_{i+1}')(x)
    #    x = MaxPooling2D(pool_size = (2,2),name=f'pool_{i+1}' )(x)

    x = Flatten()(x)
    for i,n in enumerate(fc_out_num):
        x = Dense(n,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),name=f'dense_{i}')(x)
        x = Activation('relu',name=f'dense_act_{i}')(x)

    x = Dense(int(n_classes),name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model_vgg7')
    return model 

def VGG7_v2 (in_shape, n_classes, config_params) :
    
    filters = config_params[0]
    fc_num  = config_params[1]
    cnn_flt_num = [filters, filters*2, filters*4, filters*4]
    fc_out_num = [fc_num, fc_num]

    x = x_in = Input(in_shape)

    for i in range(2):
        x = Conv2D(filters=cnn_flt_num[i], kernel_size=(3,3), 
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                    name=f'conv_{i}')(x) 
                    #padding="same", name=f'conv_{i}')(x) 
        x = BatchNormalization(name=f'bn_conv_{i}')(x)
        x = Activation('relu',name=f'conv_act_{i}')(x)
        x = MaxPooling2D(pool_size = (2,2),name=f'pool_{i}' )(x)

    for i in range(2,3,2):
        x = Conv2D(filters=cnn_flt_num[i], kernel_size=(3,3), 
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                    name=f'conv_{i}')(x) 
                    #padding="same", name=f'conv_{i}')(x) 
        x = BatchNormalization(name=f'bn_conv_{i}')(x)
        x = Activation('relu',name=f'conv_act_{i}')(x)
        x = Conv2D(filters=cnn_flt_num[i+1], kernel_size=(3,3), 
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                    name=f'conv_{i+1}')(x) 
                    #padding="same", name=f'conv_{i+1}')(x) 
        x = BatchNormalization(name=f'bn_conv_{i+1}')(x)
        x = Activation('relu',name=f'conv_act_{i+1}')(x)
        x = MaxPooling2D(pool_size = (2,2),name=f'pool_{i+1}' )(x)

    x = Flatten()(x)
    for i,n in enumerate(fc_out_num):
        x = Dense(n,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),name=f'dense_{i}')(x)
        x = Activation('relu',name=f'dense_act_{i}')(x)

    x = Dense(int(n_classes),name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model_vgg7')
    return model 



def VGG11 (in_shape, n_classes, config_params) :
    
    filters = config_params[0]
    fc_num  = config_params[1]
    cnn_flt_num = [filters, filters*2, filters*4, filters*4, filters*8, filters*8, filters*8, filters*8]
    fc_out_num = [fc_num,fc_num]

    x = x_in = Input(in_shape)

    for i in range(2):
        x = Conv2D(filters=cnn_flt_num[i], kernel_size=(3,3), 
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                    padding="same", name=f'conv_{i}')(x) 
        x = BatchNormalization(name=f'bn_conv_{i}')(x)
        x = Activation('relu',name=f'conv_act_{i}')(x)
        x = MaxPooling2D(pool_size = (2,2),name=f'pool_{i}' )(x)

    for i in range(2,8,2):
        x = Conv2D(filters=cnn_flt_num[i], kernel_size=(3,3), 
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                    padding="same", name=f'conv_{i}')(x) 
        x = BatchNormalization(name=f'bn_conv_{i}')(x)
        x = Activation('relu',name=f'conv_act_{i}')(x)
        x = Conv2D(filters=cnn_flt_num[i+1], kernel_size=(3,3), 
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                    padding="same", name=f'conv_{i+1}')(x) 
        x = BatchNormalization(name=f'bn_conv_{i+1}')(x)
        x = Activation('relu',name=f'conv_act_{i+1}')(x)
        x = MaxPooling2D(pool_size = (2,2),name=f'pool_{i+1}' )(x)

    x = Flatten()(x)
    for i,n in enumerate(fc_out_num):
        x = Dense(n,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),name=f'dense_{i}')(x)
        x = Activation('relu',name=f'dense_act_{i}')(x)

    x = Dense(int(n_classes),name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model_vgg11')
    return model 

