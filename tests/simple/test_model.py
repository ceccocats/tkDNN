import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout, ELU, Reshape, Lambda, Conv1D
from keras.layers import Bidirectional, CuDNNLSTM
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.pooling import MaxPooling2D, MaxPooling3D, AveragePooling3D
from keras.models import Sequential, Model
from keras.layers import Cropping2D
import keras.backend.tensorflow_backend as KTF
import struct
from keras.models import Sequential, Model

def bin_write(f, data):
    data = data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)

def create_model():
    x1 = Input((3, 8), name='x1')
    conv = Conv1D(4, 2)(x1)
    lstm = Bidirectional(CuDNNLSTM(5, return_sequences=True))(conv)    
    lstm2 = Bidirectional(CuDNNLSTM(5, return_sequences=False))(lstm)    
    model = Model([x1], [lstm2])
    model.summary()

    return model

if __name__ == '__main__':
    print ("DATA FORMAT: ", keras.backend.image_data_format())

    model = create_model()
    model.save("net.hdf5")

    np.random.seed(2)
    x = np.random.rand(1,1,3,8)
    r = model.predict( x[0], batch_size=1)

    r = np.array([r])
    x =  x.transpose(0, 3, 1, 2)
    #r =  r.transpose(0, 3, 1, 2)
    print("in: ", np.shape(x))
    print("out: ", np.shape(r))
    print("output: ", r.tolist())

    x = np.array(x.flatten(), dtype=np.float32)
    f = open("input.bin", mode='wb')
    bin_write(f, x)

    r = np.array(r.flatten(), dtype=np.float32)
    f = open("output.bin", mode='wb')
    bin_write(f, r)

