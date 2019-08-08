
# Imports python modules
import argparse


def get_input_args():
    """
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to create and define these 5 command line arguments. If
    the user fails to provide some or all of the 5 arguments, then the default
    values are used for the missing arguments.
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 1: that's the path to the data folder
    parser.add_argument('--data_dir', type = str, default = 'flowers',
                    help = 'path to the folder of pet images')
    # Argument 2: The model architecture to use (resnet121, or vgg16 only)
    parser.add_argument('--arch', type = str, default = 'vgg16',
                    help = 'Model architecture to use')
    # Argument 3: Number of epochs to train for
    parser.add_argument('--epochs', type = int, default = 3,
                    help = 'Number of epochs to train for')

    # Argument 4: Setting the learning Rate
    parser.add_argument('--lr', type = float, default = 0.001,
                            help = 'Number of epochs to train for')

    # Argument 5: Setting the hidden units
    parser.add_argument('--hu', type = float, default = 500,
                            help = 'Number of hidden units')

    #Assign the in_args variable to parse_args and return their values
    in_arg = parser.parse_args()

    return in_arg
