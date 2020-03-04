import argparse
import torch
import data_utils as du
import model_utils as mu

# Collect the input arguments
def process_arguments():
    ''' Collect the input arguments according to the syntax
        Return a parser with the arguments
    '''
    parser = argparse.ArgumentParser(description = 'Uses a trained network to predict the input image - flower - name')
    
    parser.add_argument('--image',
                       action='store',
                       dest='input_image_path', default='flowers/valid/5/image_05209.jpg',
                       help='File path to the input flower image')
    
    parser.add_argument('--checkpoint',
                       action='store',
                       dest='checkpoint_file_path', default='checkpoint.pth',
                       help='File path to the checkpoint file to use')
    
    parser.add_argument('--top_k',
                       action='store',
                       dest='topk', default=3, type=int,
                       help='top K most likely classes to return')
    
    parser.add_argument('--mapping',
                       action='store',
                       dest='cat_name_file', default='cat_to_name.json',
                       help='file for mapping of categories to real names')
    
    parser.add_argument('--gpu',
                       action='store_true',
                       default=False,
                       help='Use GPU. The default is CPU')
    
    return parser.parse_args()

# Get input arguments and predict a probability for the flower's name
def main():
    # Get the input arguments
    input_arguments = process_arguments()
    
    # Set the device to cuda if specified
    default_device = torch.device("cuda" if torch.cuda.is_available() and input_arguments.gpu else "cpu")
    
    # Predict
    probs, classes = mu.predict(input_arguments.input_image_path, 
                                input_arguments.checkpoint_file_path,
                                default_device,
                                input_arguments.topk)
    
    # Extract species
    species = du.extract_mapping(input_arguments.cat_name_file, classes)
    
    i = 0
    for specie in species:
        print("Flower named : " + species[i] + " predicted with probability: " + str(probs[i]))
        i += 1
    
    pass

if __name__ == '__main__':
    main()