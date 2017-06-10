from keras import layers, regularizers
from keras.models import Sequential, Model
from keras.layers.noise import GaussianNoise
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda
from keras.layers import merge, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import tensorflow as tf

def modified_VGG16(input_shape=(64, 64, 3), drop_rate=0.5, reg=0.01):
    model = Sequential()
       
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', input_shape=input_shape, data_format='channels_last', name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block1_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"))
    
    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block2_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block3_conv3'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block4_conv3'))
    #model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"))
    
    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block5_conv3')) 
    #model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"))
        
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(reg), name='fc1_timg'))
    model.add(Dropout(drop_rate))
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(reg), name='fc2_timg'))
    model.add(Dropout(drop_rate))
    model.add(Dense(200, activation='softmax', kernel_regularizer=regularizers.l2(reg), name='fc3_200'))

    return model

def modified_VGG19(input_shape=(64, 64, 3), drop_rate=0.5, reg=0.01):
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', input_shape=input_shape, data_format='channels_last', name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block1_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"))
    
    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block2_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"))

    # Block 3   
    model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block3_conv3'))
    model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block3_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block4_conv3'))
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block4_conv4'))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block5_conv3'))
    model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu', data_format='channels_last', name='block5_conv4')) 
        
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(reg), name='fc1_timg'))
    model.add(Dropout(drop_rate))
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(reg), name='fc2_timg'))
    model.add(Dropout(drop_rate))
    model.add(Dense(200, activation='softmax', kernel_regularizer=regularizers.l2(reg), name='fc3_200'))

    return model


def identity_block(input_tensor, kernel_size, filters, stage, block):
    ''' This function is a modified version of https://github.com/fchollet/deep-learning-models/releases/tag/v0.1 '''
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), padding='same', strides=(1, 1), data_format='channels_last', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', strides=(1, 1), data_format='channels_last', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), padding='same', strides=(1, 1), data_format='channels_last', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    ''' This function is a modified version of https://github.com/fchollet/deep-learning-models/releases/tag/v0.1 '''
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), padding='same', strides=strides, data_format='channels_last', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', strides=(1, 1), data_format='channels_last', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), padding='same', strides=(1, 1), data_format='channels_last', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), padding='same', strides=strides, data_format='channels_last', name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def modified_ResNet50(input_shape=(64, 64, 3), reg=0.01):
    ''' This function is a modified version of https://github.com/fchollet/deep-learning-models/releases/tag/v0.1 '''
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    img_input = Input(shape=input_shape)
  
    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), data_format='channels_last', name='conv1')(img_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((2, 2), name='avg_pool')(x) # (7, 7) in original ResNet50

    x = Flatten()(x)
    x = Dense(200, activation='softmax', kernel_regularizer=regularizers.l2(reg), name='fc200')(x)
    model = Model(img_input, x)

    return model