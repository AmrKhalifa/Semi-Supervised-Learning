import torch
import torch.nn as nn 
from torch.optim.lr_scheduler import LambdaLR

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu") 

class FCNet(nn.Module):
	def __init__(self):
		super(FCNet, self).__init__()

		self.feedforward = nn.Sequential(
					nn.Linear(in_features = 100*100, out_features = 500),
					nn.BatchNorm1d(500), 
					nn.ReLU(),
					nn.Dropout(0.2),
					
					nn.Linear(in_features = 500, out_features = 100),
					nn.BatchNorm1d(100),
					nn.ReLU(),
					nn.Dropout(0.2),

					nn.Linear(in_features = 100, out_features = 25),
					nn.BatchNorm1d(25),
					nn.ReLU(),
					nn.Dropout(0.2),

					nn.Linear(in_features = 25, out_features = 5)
			)


	def forward(self, x):
		out = self.feedforward(x)
		return out 


class CNNet(nn.Module):

	def __init__(self, out = 5):
		super(CNNet, self).__init__()

		self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3)
		nn.init.xavier_normal_(self.conv_1.weight, gain = 2**0.5)
		self.conv_2 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 2)
		nn.init.xavier_normal_(self.conv_2.weight, 2**0.5)
		self.conv_3 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3)
		nn.init.xavier_normal_(self.conv_3.weight, 2**0.5)
		self.fc_1 = nn.Linear(in_features = 11*11*8, out_features = 50)
		nn.init.xavier_normal_(self.fc_1.weight, 2**0.5)
		self.fc_2 = nn.Linear(in_features = 50, out_features = out)
		nn.init.xavier_normal_(self.fc_2.weight, 2**0.5) 

		self.pool = nn.MaxPool2d(kernel_size = 2)


	def forward(self, x):

		x = self.get_features(x)
		output = self.fc_2(x)

		return output

	def get_features(self, x):
		x = self.conv_1(x)
		x = nn.ReLU()(self.pool(x))
		b1 = nn.BatchNorm2d(16).to(device)
		x = b1(x)
		d1 = nn.Dropout(0.2).to(device)
		x  = d1(x)

		x = self.conv_2(x)
		x = nn.ReLU()(self.pool(x))
		b2 = nn.BatchNorm2d(8).to(device)
		x = b2(x)
		d2 = nn.Dropout(0.1).to(device)
		x = d2(x)

		x = self.conv_3(x)
		x = nn.ReLU()(self.pool(x))
		b3 = nn.BatchNorm2d(8).to(device)
		x = b3(x)
		d3 = nn.Dropout(0.1).to(device)
		x = d3(x)
		
		x = self.fc_1(x.view(x.size(0), -1))
		x = nn.ReLU()(x)
		d4 = nn.Dropout(0.2).to(device)
		x = d4(x)

		return x 



def train_model(model, train_data, epochs = 10, cnn = True):
	loss_trace = []
	criterion = nn.CrossEntropyLoss()
	learning_rate = .01 
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	n_epochs = epochs
	model.train()
	
	model.to(device)
	
	print("started training ...")

	for epoch in range(n_epochs):
		epoch_loss = 0.0
		if epoch % 10 ==0:
			learning_rate *= 0.9
			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		for batch in train_data:
			batch_images, batch_labels = batch

			batch_images = batch_images.to(device)
			batch_labels = batch_labels.to(device)

			if cnn == True: 
				batch_output = model(batch_images)
			else: 
			  batch_output = model(batch_images.reshape(-1, 100*100))
			
			loss = criterion(batch_output, batch_labels)
			
			optimizer.zero_grad()
			loss.backward()
			epoch_loss += loss.item()
			optimizer.step()
		
		print("the loss after processing this epoch is: ", epoch_loss)
		loss_trace.append(epoch_loss)
	print("Training completed.")
	print("=*="*20)
	return model, loss_trace


def test_model(model, test_loader, cnn = True):
	model.eval()
	model.to(device)
	correct = 0

	for batch in test_loader:
		batch_images, batch_labels = batch
		
		batch_images = batch_images.to(device)
		batch_labels = batch_labels.to(device)
			
		if cnn == True: 
			predictions = model(batch_images)
		else: 
			predictions = model(batch_images.reshape(-1, 100*100))

		predictions = predictions.data.max(1, keepdim=True)[1]
		correct += predictions.eq(batch_labels.data.view_as(predictions)).sum()

	accuracy = float(correct.item() / (len(test_loader.dataset)))

	print("The classifier accuracy is: ", 100 * accuracy)
	
	return accuracy

if __name__== '__main__':

	pass 
