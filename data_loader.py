from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import torch
import matplotlib.pyplot as plt 

data_path = '../semi_supervised_data/train_data'

train_set = torchvision.datasets.ImageFolder(
    root=data_path,
    transform = transforms.Compose([
    transforms.Resize(size = (100, 100)),
    transforms.Grayscale(),
    transforms.ToTensor()])
)

train_loader = DataLoader(train_set, batch_size = 64, shuffle = True)


if __name__ == "__main__":

	pass 
