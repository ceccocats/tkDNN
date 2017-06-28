import keras
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout, ELU
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.pooling import MaxPooling2D, MaxPooling3D
from keras.models import Sequential, Model
from keras.layers import Cropping2D
import keras.backend.tensorflow_backend as KTF
from weights_exporter import *

def dense_model():
    model = Sequential()
    model.add(Dense(256, input_shape=(1, 512)))
    model.add(ELU())
    model.add(Dense(32))
    model.add(ELU())
    model.add(Dense(2))
    
    sgd = keras.optimizers.Adam(lr=1e-4, decay=1e-8)
    model.compile(optimizer=sgd, loss="mse")
    return model

if __name__ == '__main__':

    model = dense_model()
    wg = model.get_weights()

    export_dense("dense0", wg[0], wg[1])
    export_dense("dense1", wg[2], wg[3])
    export_dense("dense2", wg[4], wg[5])

    model.set_weights(wg)
   
    X = np.random.rand(1, 512)
    i = np.array(X, dtype=np.float32)
    i.tofile("input.bin", format="f")

    print "Input: ", i
    r = model.predict( X[None, :], batch_size=1)

    print "Result: ", r
    print "Result shape: ", np.shape(r) 