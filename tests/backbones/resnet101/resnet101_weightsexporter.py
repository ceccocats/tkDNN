import torch
import urllib
from PIL import Image
from torchvision import transforms
import numpy as np
import struct 
import os

from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.autograd import Variable

from torchsummary import summary
import torch.nn as nn

from torch.jit import trace

def create_folders():
    if not os.path.exists('debug'):
        os.makedirs('debug')
    if not os.path.exists('layers'):
        os.makedirs('layers')

def bin_write(f, data):
    data =data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)

def hook(module, input, output):
    setattr(module, "_value_hook", output)

def load_ex_image(model):
    # Download an example image from the pytorch website
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    
    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    print("input_image: ",input_image.size)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    print("input_tensor: ",input_tensor.shape)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    return model, input_batch

def exp_input(model, input_batch):
    # Export the input batch 
    model(input_batch)
    i = input_batch.cpu().data.numpy()
    i = np.array(i, dtype=np.float32)
    i.tofile("debug/input.bin", format="f")
    print("input: ", i.shape)

def print_wb_output(model):
    f = None
    for n, m in model.named_modules():
        in_output = m._value_hook
        o = in_output.data.numpy()
        o = np.array(o, dtype=np.float32)
        t = '-'.join(n.split('.'))
        o.tofile("debug/" + t + ".bin", format="f")
        print('------- ', n, ' ------') 
        print("debug  ",o.shape)
        
        if not(' of Conv2d' in str(m.type) or ' of Linear' in str(m.type) or ' of BatchNorm2d' in str(m.type)):
            continue
        
        if ' of Conv2d' in str(m.type) or ' of Linear' in str(m.type):
            file_name = "layers/" + t + ".bin"
            print("open file: ", file_name)
            f = open(file_name, mode='wb')

        w = np.array([])
        b = np.array([])
        if 'weight' in m._parameters and m._parameters['weight'] is not None:
            w = m._parameters['weight'].data.numpy()
            w = np.array(w, dtype=np.float32)
            print ("    weights shape:", np.shape(w))
        
        if 'bias' in m._parameters and m._parameters['bias'] is not None:
            b = m._parameters['bias'].data.numpy()
            b = np.array(b, dtype=np.float32)
            print ("    bias shape:", np.shape(b))
        
        if 'BatchNorm2d' in str(m.type):
            b = m._parameters['bias'].data.numpy()
            b = np.array(b, dtype=np.float32)
            s = m._parameters['weight'].data.numpy()
            s = np.array(s, dtype=np.float32)
            rm = m.running_mean.data.numpy()
            rm = np.array(rm, dtype=np.float32)
            rv = m.running_var.data.numpy()
            rv = np.array(rv, dtype=np.float32)
            bin_write(f,b)
            bin_write(f,s)
            bin_write(f,rm)
            bin_write(f,rv)
            print ("    b shape:", np.shape(b))
            print ("    s shape:", np.shape(s))
            print ("    rm shape:", np.shape(rm))
            print ("    rv shape:", np.shape(rv))

        else:
            bin_write(f,w)
            if b.size > 0:
                bin_write(f,b)

        if ' of BatchNorm2d' in str(m.type) or ' of Linear' in str(m.type):
            f.close()
            print("close file")
            f = None





if __name__ == '__main__':

    model = torch.hub.load('pytorch/vision', 'resnet101', pretrained=True)
    model.eval()

    # load an example image and load it on model
    model, input_batch = load_ex_image(model)
    model.eval()
    with torch.no_grad():
        output = model(input_batch)

    # create folders debug and layers if do not exist
    create_folders()

    # add output attribute to the layers
    for n, m in model.named_modules():
        m.register_forward_hook(hook)

    # export input bin
    exp_input(model, input_batch)

    print_wb_output(model)

    with open("resnet101.txt", 'w') as f:
        for item in list(model.children()):
            f.write("%s\n" % item)

    summary(model, (3, 224, 224))
    # print(trace(model, input_batch))
