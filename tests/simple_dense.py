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


def dense_model(inp, out):
    model = Sequential()
    model.add(Dense(out, input_shape=(1, inp)))
    model.add(ELU())

    sgd = keras.optimizers.Adam(lr=1e-4, decay=1e-8)
    model.compile(optimizer=sgd, loss="mse")

    return model


if __name__ == '__main__':

    model = dense_model(8, 2)
    wg = model.get_weights()
    w = np.squeeze(wg[0])
    w = np.array([ i[0] for i in w ] + [ i[1] for i in w ], dtype=np.float32) 
    b = np.squeeze(wg[1])
    
    print "weigths: ", w
    print "bias: ", b
    w.tofile("dense.bin", format="f")
    b.tofile("dense.bias.bin", format="f")
   
    X = np.array([[[0,1,2,3,4,5,6,7]]], dtype=np.float32)
    i = np.squeeze(X[0][0])
    print "input: ", i    
    i.tofile("input.bin", format="f")
    
    r = model.predict( X, batch_size=1)
    print "Result: ", r
    print "Result shape: ", np.shape(r) 
