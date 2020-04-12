import torch
import torch.nn as nn 


use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu") 

class FCNet(nn.Module):
	def __init__(self):
		super(FCNet, self).__init__(self)

		feedforward = nn.Sequential(
					nn.Linear(n_features = 100, out_features = 50), 
					nn.ReLU(),
					nn.nn.BatchNorm1d(50),
					nn.Linear(n_features = 50, out_features = 25),
					nn.BatchNorm1d(25), 
					nn.Linear(n_features = 25, out_features = 5)
			)


	def forward(self, x):
		out = feedforward(x)
		return out 

def train_model(model, train_data, epochs = 10):
    loss_trace = []
    criterion = nn.CrossEntropyLoss()
    learning_rate = .01 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_epochs = epochs
    model.train()
    
    model.to(device)
    
    print("started training ...")

    for epoch in range(n_epochs):
    	if epoch / 10 ==0:
    		learning_rate /= 2
    		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    	epoch_loss = 0.0 
    	for batch in train_data:
    		batch_images, batch_labels = batch

    		batch_images = batch_images.to(device)
    		batch_labels = batch_labels.to(device)

    		batch_output = model(batch_images)
            
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


if __name__== '__main__':

	pass 
