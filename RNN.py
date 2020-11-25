import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# class RNN(nn.Module):
#     def __init__(self,input_size, hidden_size, num_layers, num_classes):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
#         # self.fc = nn.Linear(sequence_length*hidden_size,num_classes)
#         self.fc = nn.Linear(hidden_size,num_classes)
#     def forward(self,x):
#         h0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(device=device)
#         c0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(device=device)
#         out, _ = self.lstm(x,(h0,c0))
#         # out = out.reshape(out.shape[0],-1)
#         # out = self.fc(out)
#         out = self.fc(out[:,-1,:])
#         return out

class BRNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
        # self.fc = nn.Linear(sequence_length*hidden_size,num_classes)
        self.fc = nn.Linear(hidden_size*2,num_classes)
    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2,x.size(0), self.hidden_size).to(device=device)
        c0 = torch.zeros(self.num_layers*2,x.size(0), self.hidden_size).to(device=device)
        out, _ = self.lstm(x,(h0,c0))
        # out = out.reshape(out.shape[0],-1)
        # out = self.fc(out)
        out = self.fc(out[:,-1,:])
        return out


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#Load data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform= transforms.ToTensor(),download = True)
train_loader = DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform= transforms.ToTensor(),download = True)
test_loader = DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle=True)

#Intialize network
# model = RNN(input_size, hidden_size, num_layers, num_classes).to(device= device)
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device= device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#Train a network
for epoch in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device = device).squeeze(1)
        targets = targets.to(device =device)
        scores = model(data)
        loss = criterion(scores,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy on train data")
    else:
        print("checking acuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device).squeeze(1)
            y = y.to(device =device)
            scores = model(x)
            _,predictions = scores.max(1)
            num_correct +=(predictions == y).sum()
            num_samples+=predictions.shape[0]

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)

