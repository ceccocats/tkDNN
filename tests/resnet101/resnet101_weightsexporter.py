import torch
import urllib
from PIL import Image
from torchvision import transforms
from torchsummary import summary
import numpy as np

def hook(module, input, output):
    setattr(module, "_value_hook", output)



def print_wb(model, folder):
    for name, param in model.named_parameters():
        print ("Layer", name)
        t = name.split('.')[0:-1]
        arg = name.split('.')[-1]
        t = '-'.join(t)
        print ("    type: ", t)

        if arg == 'weight':
            w = param.data.numpy()
            print ("    weights shape:", np.shape(w))
            w.tofile(folder + "/" + t + ".bin", format="f")
        elif arg == 'bias':
            b = param.data.numpy()
            print ("    bias shape:", np.shape(b))
            b.tofile(folder + "/" + t + ".bias.bin", format="f")
        else:
            print("Ops!")


def print_wb_output(model, input_batch):
    for n, m in model.named_modules():
        m.register_forward_hook(hook)

    model(input_batch)
    i = input_batch.data.numpy()
    i = np.array(i, dtype=np.float32)
    print(i.shape)
    i.tofile("debug/input.bin", format="f")

    f = None
    for n, m in model.named_modules():
        in_output = m._value_hook
        print(n, ' ----------------------------------------------------------------')
        o = in_output.data.numpy()
        o = np.array(o, dtype=np.float32)
        t = '-'.join(n.split('.'))
        o.tofile("debug/" + t + ".bin", format="f")

        if 'Conv2d' in str(m.type) or 'Linear' in str(m.type):
            f = open("layers/" + t + ".bin", mode='wb')

        if f is None:
            continue

        # print(m._parameters)
        print(m.type)

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
        else:
            b = np.zeros(w.shape[0], dtype=np.float32)
            print ("    bias shape:", np.shape(b))
        
        if 'BatchNorm2d' in str(m.type):
            s = np.zeros(w.shape[0], dtype=np.float32)+1
            s.tofile(f, format="f")

        w.tofile(f, format="f")
        b.tofile(f, format="f")

        if 'BatchNorm2d' in str(m.type) or 'Linear' in str(m.type):
            f.close()
            f = None





if __name__ == '__main__':

    model = torch.hub.load('pytorch/vision', 'resnet101', pretrained=True)
    model.eval()

    # Download an example image from the pytorch website
    url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    print(output)


    print_wb_output(model, input_batch)

    print(list(model.children()))
