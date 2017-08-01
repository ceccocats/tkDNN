import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout, ELU, Reshape, Lambda
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.pooling import MaxPooling2D, MaxPooling3D, AveragePooling3D
from keras.models import Sequential, Model
from keras.layers import Cropping2D
import keras.backend.tensorflow_backend as KTF

def dense_model():
    model = Sequential()

    model.add(Reshape((10, 10, 1), input_shape=(10, 10)))
    model.add(Convolution2D(2, (4, 4), subsample=(2, 2), 
                           bias_initializer='random_uniform', activation="relu"))
    model.add(Convolution2D(4, (2, 2), subsample=(1, 1), 
                           bias_initializer='random_uniform', activation="relu"))
    model.add(Flatten())
    model.add(Dense(4, bias_initializer='random_uniform', activation="relu"))    
    sgd = keras.optimizers.Adam(lr=1e-4, decay=1e-8)
    model.compile(optimizer=sgd, loss="mse")
    return model


if __name__ == '__main__':
    print "DATA FORMAT: ", keras.backend.image_data_format()

    model = dense_model()
    model.save("net.h5")

    grid = np.random.rand(10,10)
    X = grid[None,:,:]
    i = np.array(grid.flatten(), dtype=np.float32)
    print i
    i.tofile("input.bin", format="f")
    print "Input: ", X    

    r = model.predict( X, batch_size=1)
    print np.shape(r)  
    print "Result: ", r
    print "Result shape: ", np.shape(r) 
    r.tofile("output.bin", format="f")
