import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties

from PIL import Image
from logger import *
import numpy as np

import data_utils as du

# Set the model to use
def set_pretrained_model(model_name):
	''' Load the model to use
		Freeze the features parameters to avoid backprop through them
		returns the model
	'''
	# Load the model
	model = getattr(models, model_name)(pretrained=True)

	# Freeze the features parameters so we don't backprop through them
	for param in model.parameters(): param.requires_grad = False
	    
	return model

# Replace the model classifier with the one specified
def set_model_classifier(model, hidden_layer, input_size=25088, output_size=2, dropout=0.5):
	''' Replace the given model classifier with the one using the specified parameters
	'''
	model.classifier = nn.Sequential(nn.Linear(input_size, hidden_layer),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_layer, output_size),
                                 nn.LogSoftmax(dim=1))

	return model

# Train the model
def train_model(model, trainloader, set_epochs, validloader, learning_rate, device,choosen_architecture):
    ''' Train the given model
	'''
    # Generate criterion
    criterion = nn.NLLLoss()
    logger = Logger("logs")

    # Train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Move the model to the available device
    model.to(device)
    
    # Set epochs
    epochs = set_epochs

    # Number of train steps, so we track it
    steps = 0

    # Track the loss, so set to zero
    running_x = [[]]
    running_y = [[]]
    running_label = [[]]
    running_loss_it = 0
    # Steps we want to go before print out the validation loss
    print_every = 1
    
    # Training, looping through the epochs
    for epoch in range(epochs):

        # looping through the data to train
        for inputs, labels in trainloader:
            # increment steps each time
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            # zero our gradients
            optimizer.zero_grad() 
            # Get our log probabilties from our model
            logps = model.forward(inputs) 
            # Get loss from the criterion on the labels
            loss = criterion(logps, labels) 
            # Backward pass
            loss.backward() 
            # Take a step with the optimizer
            optimizer.step() 
            # keep track of our training loss
            running_loss_it +=loss.item()

            # Drop out to the training loop and test the network
            # accuracy and loss on our test dataset

            # if zero, we go into the validation loop
            if steps % print_every == 0:
                accuracy = 0
                # Turn the model into evaluation inference mode which turns off dropout
                # so we can use the network to make predictions instead a test loss and accuracy
                model.eval()
                validloss = 0
                with torch.no_grad():
                    # Get images and label from the validation data
                    for inputs, labels in validloader:
                        # Transfer tensors over to the GPU
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        validloss += batch_loss
                        # Keep track of our loss to test 

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        logps = list(logps.cpu().numpy())
                        labels = list(labels.cpu().numpy())
                        running_x.extend(logps)
                        running_label.extend(labels)



    
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss_it/print_every:.3f}.. "
                      f"Validation loss: {validloss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")

                evaluation_metrics = [
                (choosen_architecture+"_Train loss",running_loss_it/print_every),
                (choosen_architecture+"_Validation loss",validloss/len(validloader)),
                (choosen_architecture+"_Validation accuracy",accuracy/len(validloader)),]
                
                logger.list_of_scalars_summary(evaluation_metrics, steps)




                # Reinit running loss
                running_loss_it = 0
                # Put the model back into training mode
                model.train()
                
    return model, epochs, optimizer

# Save the checkpoint
def create_checkpoint(model, model_name, train_data, epochs, optimizer, checkpoint_file_path, input_size=25088, output_size=102):
    ''' Create a checkpoint file for the given model
    '''
    # Get the mapping of classes to indices
    model.class_to_idx = train_data.class_to_idx
    
    # Create the checkpoint
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'epochs': epochs,
                  'model_name': model_name,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state': optimizer.state_dict(),
                  'state_dict': model.state_dict()}
    
    # Save the model
    torch.save(checkpoint, checkpoint_file_path)
    
    return 

# Load a model from a checkpoint file
def load_checkpoint(checkpoint_file_path):
    ''' Load a checkpoint file and return the model
    '''
    checkpoint = torch.load(checkpoint_file_path)
    model = getattr(models, checkpoint['model_name'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Predict the class from an image file
def predict(image_path, checkpoint, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Reload our model
    rebuilt_model = load_checkpoint(checkpoint)
    
    # Mode the model to the default device
    rebuilt_model = rebuilt_model.to(device)
    
    # Pre-process the image to use
    processed_image = du.process_image(Image.open(image_path))
    
    # Convert it: Torch tensor via array
    processed_image = torch.from_numpy(np.array(processed_image))
    
    # Add dimension to the image
    processed_image = processed_image.unsqueeze_(0)
    
    # Put model to evaluation mode
    rebuilt_model.eval()
    
    # Move the image to the available device
    processed_image = processed_image.to(device)
    
    # Turn off gradients, send image through network
    with torch.no_grad():
        output = rebuilt_model.forward(processed_image)
    
    # Get the probabilities
    probabilities = torch.exp(output)
    
    # Extract the probabilities
    probs = probabilities.topk(topk)[0]
    index = probabilities.topk(topk)[1]
    
    # Convert them to list
    probs = np.array(probs)[0]
    index = np.array(index)[0]
    
    # Now get our index, class mapping from model
    class_to_idx = rebuilt_model.class_to_idx
    
    # Invert the dictionnary
    inv_class_to_idx = {x: y for y, x in class_to_idx.items()}
    
    # Convert index to class
    classes = []
    for element in index:
        classes += [inv_class_to_idx[element]]
        
    return probs, classes

def plot(train_loss,valid_loss,accuracy):
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

    x_tran = range(0,len(train_loss),1)
    x_valid = range(0,len(valid_loss),1)
    x_acc = range(0,len(accuracy)*3,3)
    plt.figure()
    plt.plot(x_tran,train_loss,'-',color='red',label='VGG16 tarinning loss')
    plt.plot(x_valid,valid_loss,':',color = 'green',label = 'VGG16 Valid loss')
    plt.plot(x_acc,accuracy,'-.',color = 'blue',label = 'Accuracy')
    plt.title('the trainning for chicken',fontsize=24)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # plt.ylim(0, 1)
    # plt.xlim(0, 120)
    # plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    plt.title('the trainning for chicken',fontsize=24)
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Score',fontsize=20)
    plt.legend(fontsize=16)

    plt.show()

    
