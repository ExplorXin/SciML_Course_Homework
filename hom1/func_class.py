import time
import os
import sys

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False

torch.manual_seed(1234)
np.random.seed(1234)

os.environ['CUDA_VISIBLE_DEVICES']='0'
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on: %s'%(device))


from Hom1.net import FNN

# 目标函数
def target_func(x):
    return np.sin(5*np.pi*x) * np.log(x+2)

class Hom1Model:
    def __init__(self):
        self.default_epochs = 30000
        self.default_n = np.array([i for i in range(0, 
                    self.default_epochs+1, 1000)])
        self.default_100n = np.array([i for i in range(0, 
                    self.default_epochs+1, 100)])

    # 生成数据
    def generate_data(self, num_train=50):
        num_test = 151
        x = np.linspace(-1, 1, num_test).reshape((-1, 1))
        y = target_func(x)
        idx = np.random.choice(num_test, num_train, replace=False)
        
        x_train, y_train = x[idx], y[idx]
        return x_train, y_train, x, y
    
    # 模型训练
    def train_model(self, x_train, y_train, layers=[1, 20, 20, 1], actn='tanh', epochs=30000):
        model = FNN(layers, actn)
        model = model.to(device)
        
        x_train, y_train = torch.tensor(x_train, dtype=torch.float32), \
            torch.tensor(y_train, dtype=torch.float32)
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        loss_fn = nn.MSELoss()
        
        loss_history = []
        partial_loss_history = []
        
        start_time = time.time()
        
        for epoch in range(0, epochs+1):
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            # loss_history.append(loss.item())

            if epoch%100 == 0:
                loss_history.append(loss.item())
            
            if epoch % 1000 == 0:
                print('Steps: %d, loss: %.3e'%(epoch, loss.item()))
                partial_loss_history.append(loss.item())
        
        end_time = time.time()
        
        train_time = end_time - start_time
        loss_history = np.array(loss_history).reshape(-1, 1)
        return model.cpu(), loss_history, partial_loss_history, train_time
    
    # 模型评估
    def evaluate(self, model):
        x_test = np.linspace(-1, 1, 151).reshape((-1, 1))
        y_true = target_func(x_test)
        
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        
        with torch.no_grad():
            y_pred = model.cpu()(x_test_tensor)
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        return x_test, y_true, y_pred, mse, mae
    
    # 损失函数可视化
    def plot_loss(self, loss_dict, title):
        fig, ax = plt.subplots(figsize=(6,4))
        for key, loss in loss_dict.items():
            ax.plot(self.default_n, loss, label=key)
        
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5000))
        
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((4,4))
        ax.xaxis.set_major_formatter(formatter)
        
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(ls='-', alpha=0.15)
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_loss_100(self, loss, key):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(self.default_100n, loss, label=key)

        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5000))

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((4,4))
        ax.xaxis.set_major_formatter(formatter)

        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(ls='-', alpha=0.15)
        # plt.title(title)
        plt.legend()
        plt.show()
    

    # 真实-预测曲线
    def plot_curve(self, x_true, y_true, y_pred, title):
        plt.figure(figsize=(6,4))
        plt.plot(x_true, y_true, 'k-', label="True Function")
        plt.plot(x_true, y_pred, 'r--', label="Predict")
        plt.title(title)
        plt.grid(ls='-', alpha=0.15)
        plt.legend()
        plt.show()
    
    # 训练时间图
    def plot_train_time(self, dict, time_list, xlabel):
        pro = [str(s) for s in dict]
        
        plt.figure(figsize=(6,4))
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(pro)))
        bars = plt.bar(pro, time_list, width=0.6, color=colors,
                       edgecolor='black', linewidth=1)
        
        plt.xlabel(xlabel)
        plt.ylabel("Training Time (s)")
        plt.ylim(bottom=min(time_list)-5)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2,
                     height + 0.02, f"{height:.3f}", ha='center', va='bottom')
        
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model_wrapper = Hom1Model()
    
    num_train = 50
    x_train, y_train, x_ref, y_ref = model_wrapper.generate_data(num_train)
    
    plt.figure()
    plt.plot(x_train, y_train, 'bo', label='Sample Points')
    plt.plot(x_ref, y_ref, 'k-', label='True Function')
    plt.legend()
    plt.show()
    
    layers = [1] + [20]*2 + [1]
    actn = 'tanh'
    epochs = 30000
    
    model, loss, partial_loss, train_time = model_wrapper.train_model(x_train, y_train, layers, actn, epochs)
    
    x_ref = torch.tensor(x_ref, dtype=torch.float32)
    y_test = model(x_ref)
    
    plt.figure()
    plt.plot(x_train, y_train, 'bo', label='Sample Points')
    plt.plot(x_ref, y_ref, 'k-', label='True Function')
    plt.plot(x_ref, y_test.detach().numpy(), 'r--', label="Predict")
    plt.legend()
    plt.grid(ls='-', alpha=0.15)
    plt.show()
