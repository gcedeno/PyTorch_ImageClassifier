"""
Trains a new network on a dataset of images. The training loss, validation loss
and accuracy are printed out as the network trains. 
"""
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

def trainer(model,trainloader,validloader,testloader,num_epochs,learning_rate, train_data,arch):

    #Setting up the Criterion and Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    ##########Implementation of a function for the validation pass############
    def validation(model, validloader, criterion):
        ################### Testing DataLoader #########################

        #Select GPU if available otherwise run on cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #Making sure model is in evaluation mode for validation
        model.to(device)
        model.eval()
        test_loss = 0
        accuracy = 0
        #Iterating through the images in the validation dataset
        for images, labels in validloader:
            #Moving images and labels to GPU(if available)
            images, labels = images.to(device), labels.to(device)
            #Forward pass
            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return test_loss, accuracy
        ###############################


    ################# End Validation Function #######################

    ################# Training the Model ############################
    #Training the model and checking for validation loss and Accuracy
    epochs = num_epochs
    print_every = 20
    steps = 0
    model.train()
    ###### Debugging print statements ########################
    print("-------------------------------------------------------------------")
    print("Training the model .... ")
    print("-------------------------------------------------------------------")
    ## Automatic selection between between GPU if available and CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model is in evaluation mode for validation
                model.eval()

                # Turning off gradients for validation to save memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Making sure training is back on
                model.train()
    print("Done Training the Model")
    print("-------------------------------------------------------------------------------------")
    ############################ End Training the Model ##################

    ################### Testing the network ##############################
    print("Testing the Model")
    print("-------------------------------------------------------------------------------------")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()



    print('Accuracy of the model on the {}'.format(total) + ' test images: %d %%\n' % (100 * correct / total))
    print("-------------------------------------------------------------------------------------")
    #######################Done Testing the Network ###########################
    ###########################################################################
    #Save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'classifier': model.classifier,
        'optimizer': optimizer,
        'arch': arch,
        'state_dict': model.state_dict(),
        'epochs':epochs,
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint,'checkpoint.pth')

    print("checkpoint saved")
