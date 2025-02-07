
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda, MaxPool2D, GlobalAveragePooling2D

from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam, SGD

def conv2d_bn(x, filters, kernel_size, block_num,  j=0, weight_decay=.0, strides=(1, 1)):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   name='conv2d_{}_{}'.format(block_num, j),
                   kernel_initializer='lecun_uniform', #kernel_regularizer=l1(0.0001), 
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay)
                   )(x)
    layer = BatchNormalization(name=f'conv_bn_{block_num}_{j}')(layer)
    return layer

def conv2d_bn_relu(x, filters, kernel_size, block_num, j=0, weight_decay=.0, strides=(1, 1)):
    layer = conv2d_bn(x, filters, kernel_size, block_num, j, weight_decay, strides)
    layer = Activation('relu', name=f'conv_act_{block_num}_{j}')(layer)
    return layer


def ResidualBlock(x, filters, kernel_size, block_num, weight_decay,  downsample=True):
    j = 0
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, block_num=block_num, j = j, strides=2)
        j = j + 1
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              block_num = block_num,
                              j = j,
                              weight_decay=weight_decay,
                              strides=stride,
                              )
    j = j + 1
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         block_num = block_num,
                         j = j,
                         weight_decay=weight_decay,
                         strides=1,
                         )
    #out = layers.add([residual_x, residual])
    out = Add()([residual_x, residual])
    out = Activation('relu', name=f'conv_act_{block_num}_{j}')(out)
    return out

def ResNet18(input_shape, n_classes, config_params=[], weight_decay=1e-4):

    base_filters = config_params[0]
    x_in = Input(shape=input_shape)
    x = x_in
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    x = conv2d_bn_relu(x, filters=base_filters, kernel_size=(3, 3), block_num=0, weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    x = ResidualBlock(x, filters=base_filters, kernel_size=(3, 3), block_num=1, weight_decay=weight_decay, downsample=False)
    x = ResidualBlock(x, filters=base_filters, kernel_size=(3, 3), block_num=2, weight_decay=weight_decay, downsample=False)
    # # conv 3
    x = ResidualBlock(x, filters=2*base_filters, kernel_size=(3, 3), block_num=3, weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=2*base_filters, kernel_size=(3, 3), block_num=4, weight_decay=weight_decay, downsample=False)
    # # conv 4
    x = ResidualBlock(x, filters=4*base_filters, kernel_size=(3, 3), block_num=5, weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=4*base_filters, kernel_size=(3, 3), block_num=6, weight_decay=weight_decay, downsample=False)
    # # conv 5
    x = ResidualBlock(x, filters=8*base_filters, kernel_size=(3, 3), block_num=7, weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=8*base_filters, kernel_size=(3, 3), block_num=8, weight_decay=weight_decay, downsample=False)
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(n_classes, name='output_dense')(x)
    x = Activation(activation='softmax', name='output_softmax')(x)
    model = Model(inputs=[x_in], outputs=[x], name='ResNet18')

    model.compile(optimizer=SGD(lr=0.1, momentum=0.9, nesterov=False), loss=['categorical_crossentropy'], metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=args.lr, amsgrad=True), loss=['categorical_crossentropy'], metrics=['accuracy'])
    return model


def ResNet8_v2_4096(input_shape, n_classes, config_params=[], weight_decay=1e-4):

    base_filters = config_params[0]
    scale_factor = 1.36
    x_in = Input(shape=input_shape)
    x = x_in
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    x = conv2d_bn_relu(x, filters=base_filters, kernel_size=(3, 3), block_num=0, weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    x = ResidualBlock(x, filters=base_filters, kernel_size=(3, 3), block_num=1, weight_decay=weight_decay, downsample=False)
#    x = ResidualBlock(x, filters=base_filters, kernel_size=(3, 3), block_num=2, weight_decay=weight_decay, downsample=False)
    x = ResidualBlock(x, filters=scale_factor*base_filters, kernel_size=(3, 3), block_num=2, weight_decay=weight_decay, downsample=True)
    # # conv 3
    x = ResidualBlock(x, filters=scale_factor*base_filters, kernel_size=(3, 3), block_num=3, weight_decay=weight_decay, downsample=True)
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(n_classes, name='output_dense')(x)
    x = Activation(activation='softmax', name='output_softmax')(x)
    model = Model(inputs=[x_in], outputs=[x], name='ResNet18')

    model.compile(optimizer=SGD(lr=0.1, momentum=0.9, nesterov=False), loss=['categorical_crossentropy'], metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=args.lr, amsgrad=True), loss=['categorical_crossentropy'], metrics=['accuracy'])
    return model


def ResNet9_4096(input_shape, n_classes, config_params=[], weight_decay=1e-4):

    base_filters = config_params[0]
    scale_factor = 1.36
    x_in = Input(shape=input_shape)
    x = x_in
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    x = conv2d_bn_relu(x, filters=base_filters, kernel_size=(3, 3), j=0, block_num=0, weight_decay=weight_decay, strides=(1, 1))
    x = conv2d_bn_relu(x, filters=base_filters, kernel_size=(3, 3), j=1, block_num=0, weight_decay=weight_decay, strides=(1, 1))
    x = MaxPooling2D(pool_size = (2,2), padding="same", name='pool_0' )(x)
    # # conv 2
    x = ResidualBlock(x, filters=base_filters, kernel_size=(3, 3), block_num=1, weight_decay=weight_decay, downsample=False)
#    x = ResidualBlock(x, filters=base_filters, kernel_size=(3, 3), block_num=2, weight_decay=weight_decay, downsample=False)
    x = ResidualBlock(x, filters=scale_factor*base_filters, kernel_size=(3, 3), block_num=2, weight_decay=weight_decay, downsample=True)
    # # conv 3
    x = ResidualBlock(x, filters=scale_factor*base_filters, kernel_size=(3, 3), block_num=3, weight_decay=weight_decay, downsample=True)
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(n_classes, name='output_dense')(x)
    x = Activation(activation='softmax', name='output_softmax')(x)
    model = Model(inputs=[x_in], outputs=[x], name='ResNet18')

    model.compile(optimizer=SGD(lr=0.1, momentum=0.9, nesterov=False), loss=['categorical_crossentropy'], metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=args.lr, amsgrad=True), loss=['categorical_crossentropy'], metrics=['accuracy'])
    return model


def residual_conv_block(filters, down_sample=False):
    def layer(input_tensor):

        res = input_tensor
        strides = [2, 1] if down_sample else [1, 1]
        x = Conv2D(filters, strides=strides[0],
                    kernel_size=(3, 3), padding="same", 
                    kernel_initializer="he_normal")(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, strides=strides[1],
                    kernel_size=(3, 3), padding="same", 
                    kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        
        if down_sample:
            res = Conv2D(filters, strides=2, kernel_size=(1, 1), 
                            kernel_initializer="he_normal", padding="same")(res)
            res = BatchNormalization()(res)

        x = Add()([x, res])
        x = Activation('relu')(x)
        return x

    return layer


def ResNet8_v4096 (in_shape, n_classes,config_params=[]) :

    #keras.backend.set_image_data_format('channels_last')
    filters = config_params[0]
    scale_factor = 1.36
    x = x_in = Input(in_shape)
    #img_input = keras.Input(shape=[32,32,3])

    x = Conv2D(filters=filters, kernel_size=(7,7), strides=2,
                kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                padding="same", name='conv_0')(x) 


    x = BatchNormalization(name='bn_conv_0')(x)
    x = Activation('relu',name=f'conv_act_0')(x)
    #x = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)
    x = MaxPooling2D(pool_size = (2,2), padding="same", name='pool_0' )(x)
    for res_block in [residual_conv_block(filters), 
                      residual_conv_block(int(filters * scale_factor), down_sample=True),  
                      residual_conv_block(int(filters * scale_factor), down_sample=True), 
                      ]:
            x = res_block(x)

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(int(n_classes),name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model_lenet5')

    return model

def ResNet8_v8192 (in_shape, n_classes,config_params=[]) :

    #keras.backend.set_image_data_format('channels_last')
    filters = config_params[0]
    x = x_in = Input(in_shape)
    #img_input = keras.Input(shape=[32,32,3])

    x = Conv2D(filters=filters, kernel_size=(7,7), strides=2,
                kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                padding="same", name='conv_0')(x) 


    x = BatchNormalization(name='bn_conv_0')(x)
    x = Activation('relu',name=f'conv_act_0')(x)
    #x = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)
    x = MaxPooling2D(pool_size = (2,2), padding="same", name='pool_0' )(x)
    for res_block in [residual_conv_block(filters), 
                      residual_conv_block(int(filters * 1.5), down_sample=True),  
                      residual_conv_block(int(filters * 1.5), down_sample=True), 
                      ]:
            x = res_block(x)

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(int(n_classes),name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model_resnet')

    return model

def ResNet8_v2 (in_shape, n_classes,config_params=[]) :

    #keras.backend.set_image_data_format('channels_last')
    filters = config_params[0]
    x = x_in = Input(in_shape)
    #img_input = keras.Input(shape=[32,32,3])

    x = Conv2D(filters=filters, kernel_size=(7,7), strides=2,
                kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                padding="same", name='conv_0')(x) 


    x = BatchNormalization(name='bn_conv_0')(x)
    x = Activation('relu',name=f'conv_act_0')(x)
    #x = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)
    x = MaxPooling2D(pool_size = (2,2), padding="same", name='pool_0' )(x)
    for res_block in [residual_conv_block(filters), 
                      residual_conv_block(filters * 2, down_sample=True),  
                      residual_conv_block(filters * 4, down_sample=True), 
                      ]:
            x = res_block(x)

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(int(n_classes),name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model_resnet')

    return model




def ResNet10 (in_shape, n_classes,config_params=[]) :

    #keras.backend.set_image_data_format('channels_last')
    filters = config_params[0]
    x = x_in = Input(in_shape)
    #img_input = keras.Input(shape=[32,32,3])

    x = Conv2D(filters=filters, kernel_size=(7,7), strides=2,
                kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
                padding="same", name='conv_0')(x) 


    x = BatchNormalization(name='bn_conv_0')(x)
    x = Activation('relu',name=f'conv_act_0')(x)
    #x = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)
    x = MaxPooling2D(pool_size = (2,2), padding="same", name='pool_0' )(x)
    for res_block in [residual_conv_block(filters), 
                      residual_conv_block(filters * 2, down_sample=True),  
                      residual_conv_block(filters * 4, down_sample=True), 
                      residual_conv_block(filters * 8, down_sample=True), 
                      ]:
            x = res_block(x)

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(int(n_classes),name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model_resnet')

    return model

#def ResNet18 (in_shape, n_classes,config_params=[]) :
#
#    #keras.backend.set_image_data_format('channels_last')
#    filters = config_params[0]
#    x = x_in = Input(in_shape)
#    #img_input = keras.Input(shape=[32,32,3])
#
#    x = Conv2D(filters=filters, kernel_size=(7,7), strides=2,
#                #kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), 
#                kernel_initializer='he_normal', 
#                padding="same", name='conv_0')(x) 
#
#
#    x = BatchNormalization(name='bn_conv_0')(x)
#    x = Activation('relu',name=f'conv_act_0')(x)
#    #x = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)
#    x = MaxPooling2D(pool_size = (2,2), padding="same", name='pool_0' )(x)
#    for res_block in [residual_conv_block(filters), 
#                      residual_conv_block(filters), 
#                      residual_conv_block(filters * 2, down_sample=True),  
#                      residual_conv_block(filters * 2), 
#                      residual_conv_block(filters * 4, down_sample=True), 
#                      residual_conv_block(filters * 4), 
#                      residual_conv_block(filters * 8, down_sample=True), 
#                      residual_conv_block(filters * 8)]:
#            x = res_block(x)
#
#
#
#    x = GlobalAveragePooling2D()(x)
#    x = Flatten()(x)
#    x = Dense(int(n_classes),name='output_dense')(x)
#    x_out = Activation('softmax',name='output_softmax')(x)
#
#    model = Model(inputs=[x_in], outputs=[x_out], name='origin_model_resnet18')
#
#    return model



