import json
import torch
from torchvision import datasets, transforms, models


# Process images in the same manner used for training
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    processing = transforms.Compose([transforms.Resize(256), 
                                     transforms.CenterCrop(224), 
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])])
    
    image = processing(image)
   
    return image


# Load all the data
def loading_data(data_dir):
	''' Use torchvision to load training, validation and test data
		Training data: andom scaling, cropping, flipping, resized data to 224x224
		Testing and validation data: resize and crop to the appropriate size
        returns datasets and data loaders
    '''
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/test'
	test_dir = data_dir + '/test'

	# training data: random scaling, cropping, flipping, resized data to 224x224
	train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

	# testing and validation: resize and crop to the appropriate size
	test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

	# Load the datasets with ImageFolder
	train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
	test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
	valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

	# Define the dataloaders
	trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
	validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

	return train_data, test_data, valid_data, trainloader, testloader, validloader

# Extract flowers names mapping
def extract_mapping(mapping_file_path, classes):
    ''' Build a mapping dictionnary from a json mapping file
        Build a most likely species names from provided classes
        return the species
    '''
    with open(mapping_file_path, 'r') as f:
        cat_to_name = json.load(f)
    
    species = []
    
    for i in classes:
        species += [cat_to_name[i]]
    
    return species
