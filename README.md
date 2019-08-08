# Description
**AI Programming with Python Project:** This repository contains code and associated files corresponding to Udacity's final project for the AI Programming with Python Nanodegree program. This project consist of two parts, the first part requires developing code in a Jupyter Notebook to built and image classifier using PyTorch and then train it to recognize and classify flowers into different categories. This trained image classifier will be later used to recognize (predict) different species of flowers.

After building and training a deep neural network on the flower dataset, for the second part of the project, the goal is to create an application that others can use. This application consist in Python scripts that run from the command line. The main scripts are the `train.py` and `predict.py`. The first file `train.py`, trains a new neural network on a dataset and saves the model as a checkpoint. The second file, `predict.py`, uses a trained network to predict the class for an input image.

## Usage Description
## `train.py`
**PURPOSE:** Build an application to train a deep neural network on the flower dataset, from IMAGENET. The application consist of a pair of Python scripts, one for training and the other for making prediction using the trained model. Several additional scripts (helpers) are used for specific purposes and are call from the train (main) script. There are two main files as part of this application, namely: train.py and predict.py. The first file, train.py, trains a new network on a dataset and saves the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image.

The training script has been implemented with only two different architectures taken from torchvision.models: vgg16 and densenet121

**Program automatically runs on GPU if available, on cpu otherwise**

This application, train.py, trains a new network on a dataset and saves the trained
model as a checkpoint for later with predict.py application.
The training script has been implemented with only two different architectures
taken from torchvision.models: vgg16 and densenet121

#### Usage Example
Use argparse Expected Call with <> indicating expected user input:
     python train.py --data_dir <directory with images> --arch <model> (vgg16 or densenet121 only)
            --epochs <num. training epochs> --lr <learning rate> --hu <hidden units>
            --to_device <select running on 'gpu' or 'cpu'>
 Example call:

  python train.py --data_dir flower_data --arch vgg16 --epochs 3 --lr 0.001 --hu 500 --to_device 'gpu'

When calling "train.py" without input arguments, program runs using default values

## `predict.py`
**Program automatically runs on GPU if available, on cpu otherwise**

The predict.py script reads in an image and a checkpoint,  then prints the most likely image class and it's associated probability.
The script prints out the top k classes along with associated probabilities. Loads a JSON file that maps the class values to category name.

#### Usage Example
Use argparse Expected Call with <> indicating expected user input:

     python predict.py --chkp_path <path to the saved checkpoint>
     --img_path <The path for the image to be used for prediction>
     --topk <Number of top k flower classes>
     --json_path <path to the JSON file for Label Mapping>
     --to_device <run model on CPU or GPU>

When calling "predict.py" without input arguments, program runs using default values
# PyTorch_ImageClassifier
