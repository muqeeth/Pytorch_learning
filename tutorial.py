import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
mytensor = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.float32, device= device,\
    requires_grad= True)
 
# print(mytensor)
# print(mytensor.device)
# print(mytensor.shape)

x = torch.empty(size=(3,3))
x = torch.zeros((3,3))
x = torch.ones((3,3))
x = torch.rand((3,3))
x = torch.eye(3,3)
x = torch.arange(0,5,1)
x = torch.linspace(0.1,1,10)
x = torch.empty(size = (3,3)).normal_(mean = 0, std = 1)
x = torch.diag(torch.ones(3))

tensor = torch.arange(4)
# print(tensor.bool())
# print(tensor.float())
# print(tensor.double())

import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()


x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])
z1 = torch.empty(3)
torch.add(x,y,out = z1)
# print(z1)

z2 = torch.add(x,y)
z = x+y
z = x-y
z = torch.true_divide(x,y)

t = torch.zeros(3)
t.add_(x)
t+=x ## t= t+x is bad

z = x.pow(2)
z = x**2

z = x>0

x1 = torch.rand((2,5))
x2 = torch.rand(5,3)
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

matrix_ep = torch.rand((3,3))
# print(matrix_ep.matrix_power(3))

z = x*y
z = torch.dot(x,y)

batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1,tensor2)

x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
z = x1 - x2
z = x1**x2
# print(z)

sum_x = torch.sum(x,dim = 0)
values,indices = torch.max(x,dim = 0)
# print(values,indices)
abs_x = torch.abs(x)
z = torch.argmax(x,dim = 0)

mean_x = torch.mean(x.float(),dim = 0)
z = torch.eq(x,y)
sorted_y,indices = torch.sort(y,dim = 0, descending= False)

z = torch.clamp(x, min = 0, max= 10)
x = torch.tensor([1,0,1,1,1], dtype= torch.bool)
z = torch.any(x)
z = torch.all(x)
# print(z)

batch_size = 10
features = 25
x = torch.rand((batch_size,features))
# print(x[0,:].shape)
# print(x[:,0].shape)
# print(x[2,0:10])

x = torch.arange(10)
indices = [1,2,5]
# print(x[indices])

x = torch.arange(10)
# print(x[(x>2) & (x<8)])
# print(x[x.remainder(2) == 0])


# print(torch.where(x>5,x , x*2))
# print(torch.tensor([1,1,1,2]).unique())
# print(x.ndimension())
# print(x.numel())

x = torch.arange(9)
x_3x3 = x.view(3,3)
# print(x_3x3.shape)
x_3x3 = x.reshape(3,3)
y = x_3x3.t()
# print(y.contiguous().view(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
# print(torch.cat((x1,x2), dim = 0).shape)
# print(torch.cat((x1,x2), dim = 1).shape)

z = x1.view(-1)
batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch,-1)

z = x.permute(0,2,1)
x = torch.arange(10)
# print(x.unsqueeze(0).shape)
# print(x.unsqueeze(1).shape)
x = torch.arange(10).unsqueeze(0).unsqueeze(1)
# print(x.shape)
z = x.squeeze(1)
# print(z.shape)