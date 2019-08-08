"""
### Program automatically runs on GPU if available, on cpu otherwise ########

The predict.py script reads in an image and a checkpoint,  then prints the most
likely image class and it's associated probability.
The script prints out the top k classes along with associated probabilities.
Loads a JSON file that maps the class values to category name.

########### Using the application #################################
Use argparse Expected Call with <> indicating expected user input:

     python predict.py --chkp_path <path to the saved checkpoint>
     --img_path <The path for the image to be used for prediction>
     --topk <Number of top k flower classes>

When calling "train.py" without input arguments, program runs using default values
"""
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
####################
import argparse


################################### Input Arguments #############################

def get_input_args():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 1: that's the path to the checkpoint
    parser.add_argument('--chkp_path', type = str, default = 'checkpoint.pth',
                    help = 'path to the saved checkpoint')
    # Argument 2: The path for the image used for prediction
    parser.add_argument('--img_path', type = str, default = 'flowers/test/1/image_06743.jpg',
                    help = 'The path for the image used for prediction')
    # Argument 3: Number of Topk most probable classes
    parser.add_argument('--topk', type = int, default = 5,
                    help = 'Number of top k flower classes')

    #Assign the in_args variable to parse_args and return their values
    in_arg = parser.parse_args()

    return in_arg

in_arg = get_input_args()
################################### Label Mapping #############################

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# ############ Sanity Checking ###################################

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint= torch.load(filepath, map_location=lambda storage, loc: storage)

    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer = optim.Adam(model.classifier.parameters())#, lr=learning_rate)
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    #optimizer = optim.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint['epochs']
    #Check if GPU is available, otherwise run on cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #Run on evaluation mode for inference
    model.eval()

    print("checkpoint loaded")

    return model

#loading the model and checking it
model = load_checkpoint(in_arg.chkp_path)
#print("Model Classifier loaded for predictions: {}".format(model.classifier))

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    #Formatting the images as used for training
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    IM = Image.open(image)
    pil_image = preprocess(IM).float()
    np_image = np.array(pil_image) #converting color channels to 0-1 floats


    #Normalization
    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    #Reordering dimensions using ndarray.transpose
    norm_image = (np.transpose(np_image, (1, 2, 0)) - means)/std
    proc_image = np.transpose(norm_image, (2, 0, 1))

    ### OBS: function returns an np.array ##########
    return proc_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    #ax.imshow(image)


    return image #ax

#Checking preprocessing
#imshow(process_image('flower_data/test/1/image_06743.jpg'))

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Making sure model is in evaluation mode for validation
    model.to(device)
    #model.eval()

    #Calling the preprocessing function
    proc_image = process_image(image_path)
    # process_image function returns a np.array(), conversion needed before using it on the model
    t_proc_image = torch.from_numpy(proc_image)
    torch_image = t_proc_image.unsqueeze_(0)
    torch_image = torch_image.type(torch.FloatTensor)
    torch_image = torch_image.to(device)
    output = model(torch_image)
    ######### Correction for negative probabilities #################################
    funct_ps = torch.exp(output)
    #getting the topk largest values from the tensor (default = 5)
    probs, top_classes = torch.topk(funct_ps, topk)

    ###Changing dtypes and inverting the dictionary to get a mapping from index to class #####
    probs = [float(prob) for prob in probs[0]]

    inv_map = {v: k for k, v in model.class_to_idx.items()}

    top_classes = [inv_map[int(index)] for index in top_classes[0]]

    return probs, top_classes, topk

################### Sanity Checking ######################

path = (in_arg.img_path)

probs, classes, topk = predict(path, model, in_arg.topk) #############************
names = [cat_to_name[str(index)] for index in classes]
print("------------------------------------------------------------------------")
print("-------------------- Prediction Results --------------------------------\n")
print("The most probable classification for this flower is: {}".format(names[0]))
print("With an associated probability of: {:.3f}\n".format(probs[0]))
print("The top {} classes for this flower are: {}".format(topk,names))
print("With associated probabilities of: {}\n".format(probs))
print("------------------------------------------------------------------------")

proc_image = process_image(path)
max_index = classes[0]

fig = plt.figure(figsize=(6,6))
##### Showing flower image ###########
fig.add_subplot(211)
imgplot = imshow(proc_image, ax = plt)
plt.title(cat_to_name[str(max_index)])
plt.axis('off')
plt.imshow(imgplot)

### showing the graph #########
fig.add_subplot(212)
y_pos = np.arange(len(names))
probabilities = np.array(probs)
plt.barh(y_pos, probabilities, align='center',
        color='blue')
plt.yticks(y_pos, names)
plt.gca().invert_yaxis()
plt.show()
