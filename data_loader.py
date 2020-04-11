from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

t = torch.tensor([1., 2., 4.], requires_grad = True)
z = torch.sum(t)

z.backward()
print(t.grad)

