import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
from torch import nn
from collections import OrderedDict
import torch

densenet121 = models.densenet121(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'densenet121': densenet121, 'vgg16': vgg16}

def classifier(model_name,hidden_units):

    # apply model to input
    model = models[model_name]

    if model_name == "vgg16":
        """ VGG16
        """
        print("Model VGG16 Selected")
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        #Feedforward Classifier using ReLU activations and dropout
        #Classifier made for a VGG16 model architecture, using three layers and a log softmax output
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, 4096)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(4096, hidden_units)),
                                  ('relu1', nn.ReLU()),
                                  ('fc3', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1)),
                                  ('dropout',nn.Dropout(p=0.3))
                                  ]))

        model.classifier = classifier

    elif model_name == "densenet121":
        """ DenseNet
        """

        print("Model DenseNet121 Selected")
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False


        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1)),
                                  ('dropout',nn.Dropout(p=0.3))
                                  ]))

        model.classifier = classifier

    #Automatic selection  between GPU and CPU. The program will run on GPU
    #unless is not available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model
