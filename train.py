
# PROGRAMMER: Gustavo Cedeno
# DATE CREATED:25.02.2019

# PURPOSE: Build an application to train a deep neural network on the flower dataset,
#          from IMAGENET. The application consist of a pair of Python scripts, one
#          for training and the other for making prediction using the trained model.
#          Several additional scripts (helpers) are used for specific purposes and are call
#          from the train (main) script.
#          There are two main files as part of this application, namely: train.py and predict.py.
#          The first file, train.py, trains a new network on a dataset and saves the
#          the model as a checkpoint. The second file, predict.py, uses a trained network
#           to predict the class for an input image.
#
# The training script has been implemented with only two different architectures
# taken from torchvision.models: vgg16 and densenet121
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --dir <directory with images> --arch <model> (vgg16 or densenet121 only)
#             --epochs <num. training epochs> --lr <learning rate> --hu <hidden units>
#   Example call showing default values:
#    python train.py --dir flower_data --arch vgg16 --epochs 3 --lr 0.001 --hu 500
##
# When calling "train.py" without input arguments, program runs using default values
"""
### Program automatically runs on GPU if available, on cpu otherwise ########

This application, train.py, trains a new network on a dataset and saves the trained
model as a checkpoint for later with predict.py application.
The training script has been implemented with only two different architectures
taken from torchvision.models: vgg16 and densenet121

########### Using the application #################################
Use argparse Expected Call with <> indicating expected user input:
     python train.py --dir <directory with images> --arch <model> (vgg16 or densenet121 only)
            --epochs <num. training epochs> --lr <learning rate> --hu <hidden units>
 Example call (showing default values):
  python train.py --dir flower_data --arch vgg16 --epochs 3 --lr 0.001 --hu 500

When calling "train.py" without input arguments, program runs using default values
"""

################# Imports python modules and Packages###################
from time import time, sleep

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
# Imports functions created for this program
from print_functions import *
from get_input_args import get_input_args
from data_loader import data_loader
from classifier import classifier
from model_trainer import *

# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    #Getting command line arguments
    in_arg = get_input_args()
    # Function that checks and print out command line arguments for training
    check_command_line_arguments(in_arg)
    # This function loads the data, define the transforms, and returns
    # the data loaders for training, valiation and testing
    trainloader,validloader,testloader,train_data =data_loader(in_arg.data_dir)
    # Selecting the model and setting up the classifier
    model = classifier(in_arg.arch,in_arg.hu)
    #Function that trains the model and runs validation and testing showing
    #the results for test_loss and accuracy. Saves the checkpoint
    trainer(model,trainloader,validloader,testloader,in_arg.epochs,in_arg.lr,train_data,in_arg.arch)

    # Measure total program runtime (Useful for comparing GPU vs CPU)
    end_time = time()
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
# Call to main function to run the program
if __name__ == "__main__":
    main()
