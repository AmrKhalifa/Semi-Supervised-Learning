from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import torch
import matplotlib.pyplot as plt 

train_data_path = '../semi_supervised_data/train_data'
test_data_path = '../semi_supervised_data/test_data'
def get_set(path): 

	dataset = torchvision.datasets.ImageFolder(
	    root=path,
	    transform = transforms.Compose([
	    transforms.Resize(size = (100, 100)),
	    transforms.Grayscale(),
	    transforms.ToTensor()])
	)

	return dataset

train_set = get_set(train_data_path)
test_set = get_set(test_data_path)

train_loader = DataLoader(train_set, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 64, shuffle = True)

if __name__ == "__main__":

	pass 
