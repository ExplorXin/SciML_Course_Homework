import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot
import time
import scipy.io as io
import matplotlib.pyplot as plt

torch.manual_seed(1234)
np.random.seed(1234)

from hom1.ReferenceCode.net import FNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on: %s'%(device))

# 目标函数
def target_func(x):
    return x**2 + np.sin(8*np.pi*x+np.pi/10)**2

# 生成数据
def generate_data():
    num_test, num_train = 101, 50
    x = np.linspace(-1, 1, num_test).reshape((-1, 1))
    y = target_func(x)
    idx = np.random.choice(num_test, num_train,  replace=False)
    x_train, y_train = x[idx], y[idx]
    return x_train, y_train, x, y


if __name__ == '__main__':

    x_train, y_train, x_ref, y_ref = generate_data()
    x_train, y_train = torch.tensor(x_train, dtype=torch.float32), \
        torch.tensor(y_train, dtype=torch.float32)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_ref

    x_ref = torch.tensor(x_ref, dtype=torch.float32)

    layers = [1] + [20]*2 + [1] #更改网络结构层数与深度

    model = FNN(layers)

    '''
    print(model.state_dict()['linear.2.weight'].shape)
    '''

    model = model.to(device)

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    nmax = 10000  # 更改训练次数
    n = 0

    ls1=[]
    ls2=[]

    while n <= nmax:
        n += 1
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)

        opt.zero_grad() #梯度清零
        loss.backward()
        opt.step()

        if n%1000 == 0:
            ls1.append(n)
            ls2.append(loss)
            print('Steps: %d, loss: %.3e'%(n, loss.item()))

    index, Loss = torch.tensor(ls1),torch.tensor(ls2)
    plt.figure()
    plt.plot(index.cpu().numpy(),Loss.cpu().numpy(),label="Train cost")
    plt.show

    y_test = model.cpu()(x_ref)

    plt.figure()
    plt.rcParams['font.sans-serif']=['KaiTi'] 
    plt.plot(x_train.cpu().numpy(), y_train.cpu().numpy(), 'bo')
    plt.plot(x_ref, y_ref, 'k-')
    plt.plot(x_ref, y_test.detach().numpy(), 'r--',label="Predict")
    plt.show()
