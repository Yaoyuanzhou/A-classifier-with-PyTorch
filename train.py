import argparse
import torch
import os
import data_utils as du
import model_utils as mu
import logger

# Collect the input arguments
def process_arguments():
    ''' Collect the input arguments according to the syntax
        Return a parser with the arguments
    '''
    parser = argparse.ArgumentParser(description = 'Train new netword on a dataset and save the model')
    
    parser.add_argument('--data_directory',
                       action='store',
                       default='D:\dataset\chickens\classfier/',
                       help='Input directory for training data')
    
    parser.add_argument('--save_dir',
                       action='store',
                       dest='save_directory', default='checkpoint_dir',
                       help='Directory where the checkpoint file is saved')
    
    parser.add_argument('--arch',
                       action='store',
                       dest='choosen_archi', default='vgg16',
                       help='Choosen models to train chicken')
    
    parser.add_argument('--learning_rate',
                       action='store',
                       dest='learning_rate', type=float, default=0.001,
                       help='Neural Network learning rate')
    
    parser.add_argument('--hidden_units',
                       action='store',
                       dest='hidden_units', type=int, default=512,
                       help='Number of hidden units')
    
    parser.add_argument('--epochs',
                       action='store',
                       dest='epochs', type=int, default=5,
                       help='Number of Epochs for the training')
    
    parser.add_argument('--gpu',
                       action='store_true',
                       default=True,
                       help='Use GPU. The default is CPU')
    
    return parser.parse_args()

# Get input arguments and train the specified network
def main():
    
    # Get the input arguments
    input_arguments = process_arguments()
    
    # Set the device to cuda if specified
    default_device = torch.device("cuda" if torch.cuda.is_available() and input_arguments.gpu else "cpu")
    
    # Set input_size for network, by default
    
    input_size = 9216
    choosen_architecture = input_arguments.choosen_archi
    
    if choosen_architecture[:3] == "vgg":
        input_size = 25088
    if choosen_architecture[:8] == "densenet":
        input_size = 1024
    
    

        
    # Load data
    train_data, test_data, valid_data, trainloader, testloader, validloader = du.loading_data(input_arguments.data_directory)
    
    # Set the choosen pretrained model
    model = mu.set_pretrained_model(choosen_architecture)
    
    # Set model classifier
    model = mu.set_model_classifier(model, input_arguments.hidden_units, input_size, output_size=2, dropout=0.05)
    
    # Train the model
    model, epochs, optimizer = mu.train_model(model, 
                                              trainloader, 
                                              input_arguments.epochs, 
                                              validloader, 
                                              input_arguments.learning_rate, 
                                              default_device,choosen_architecture)
    
    # Create a file path using the specified save_directory
    # to save the file as checkpoint.pth under that directory
    if not os.path.exists(input_arguments.save_directory):
        os.makedirs(input_arguments.save_directory)
    checkpoint_file_path = os.path.join(input_arguments.save_directory, choosen_architecture+"_"+str(input_arguments.epochs)+".pth")
    
    # Store the trained model as checkpoint
    mu.create_checkpoint(model, 
                         input_arguments.choosen_archi,
                         train_data, 
                         epochs, 
                         optimizer, 
                         checkpoint_file_path,
                         input_size,
                         output_size=2)
    
    pass
    # mu.plot(train_loss,valid_loss,accuracy_loss)

if __name__ == '__main__':
    main()