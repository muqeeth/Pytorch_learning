import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision

#create a  fullyconnected network
class CNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels = 8, kernel_size= (3,3), stride = (1,1), padding= (1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels = 16, kernel_size= (3,3), stride = (1,1), padding= (1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
        self.initialize_weights()

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias)

import sys
model = CNN()
sys.exit()

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("==> saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint):
    print('==>loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
# model = CNN()
# x = torch.randn(64,1,28,28)
# print(model(x).shape)
# exit()

    
# model = NN(784,10)
# x = torch.randn((64, 784))
# print(model(x).shape)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3
load_model = True

#Load data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform= transforms.ToTensor(),download = True)
train_loader = DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform= transforms.ToTensor(),download = True)
test_loader = DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle=True)

#Intialize network
model = CNN(in_channels, num_classes).to(device= device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

#Train a network
for epoch in range(num_epochs):
    # loop = tqdm(enumerate(train_loader),total = len(train_loader), leave = False)
    if epoch == 2:
        checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device =device)
        scores = model(data)
        loss = criterion(scores,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loop.set_description(f"Epoch[{epoch}/{num_epochs}]")
        # loop.set_postfix(loss = loss.item(), acc = torch.randn(1).item())

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
            x = x.to(device = device)
            y = y.to(device =device)
            scores = model(x)
            _,predictions = scores.max(1)
            num_correct +=(predictions == y).sum()
            num_samples+=predictions.shape[0]

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()

# check_accuracy(train_loader,model)
# check_accuracy(test_loader,model)