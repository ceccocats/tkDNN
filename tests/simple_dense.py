import keras
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout, ELU, Reshape, Lambda
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.pooling import MaxPooling2D, MaxPooling3D, AveragePooling3D
from keras.models import Sequential, Model
from keras.layers import Cropping2D
import keras.backend.tensorflow_backend as KTF
from weights_exporter import *

def dense_model():
    model = Sequential()
    model.add(Reshape((100, 100, 4, 1), input_shape=(100, 100, 4)))
    model.add(Lambda(lambda x: 2*x - 1.,
            batch_input_shape=(1, 100, 100, 4),         # 100by100by2
            output_shape=(100, 100, 4, 1)))                         # 100by100by2
    model.add(Convolution3D(16, kernel_size=(8, 8, 2), subsample=(4, 4, 1), border_mode="valid", 
                bias_initializer="random_uniform"))
    model.add(ELU())
    model.add(AveragePooling3D(pool_size=(2, 2, 1)))
    model.add(Convolution3D(16, kernel_size=(4, 4, 2), subsample=(2, 2, 1), border_mode="valid", 
                bias_initializer="random_uniform"))
    model.add(ELU())
    model.add(Convolution3D(24, kernel_size=(3, 3, 2), subsample=(1, 1, 1), border_mode="valid", 
                bias_initializer="random_uniform"))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(256, bias_initializer="random_uniform"))
    model.add(ELU())
    model.add(Dense(32, activation="relu", bias_initializer="random_uniform"))
    model.add(Dense(2, bias_initializer="random_uniform"))

    sgd = keras.optimizers.Adam(lr=1e-4, decay=1e-8)
    model.compile(optimizer=sgd, loss="mse")
    return model


if __name__ == '__main__':
    print "DATA FORMAT: ", keras.backend.image_data_format()

    model = dense_model()
    wg = model.get_weights()
    export_conv3d("conv0",  wg[0], wg[1])
    export_conv3d("conv1",  wg[2], wg[3])
    export_conv3d("conv2",  wg[4], wg[5])
    export_dense ("dense3", wg[6], wg[7])
    export_dense ("dense4", wg[8], wg[9])
    export_dense ("dense5", wg[10], wg[11])

    grid = np.random.rand(100, 100,4)
    X = grid[None,:,:]
    i = np.array(grid.flatten(), dtype=np.float32)
    print i
    i.tofile("input.bin", format="f")
    print "Input: ", X    

    r = model.predict( X, batch_size=1)
    print np.shape(r)  
    print "Result: ", r
    print "Result shape: ", np.shape(r) 