import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from Hom1.func_class import Hom1Model
model = Hom1Model()



## ---------训练点数影响---------
# 训练点数
# sample_sizes = [15, 35, 55, 75, 100]
sample_sizes = [55]

# 训练点数-MSE-MAE对
results_q1 = []
# 训练点数-损失值对
loss_dict_q1 = {}

for n in sample_sizes:
    print('当前训练点数：',n)
    # 模型训练
    x_train_q1, y_train_q1, x_ref_q1, y_ref_q1 = model.generate_data(n)
    model_q1, loss_q1, partial_loss_q1, train_time_q1 = model.train_model(x_train_q1, y_train_q1)

    # 模型评估
    x_test_q1, y_true_q1, y_pred_q1, mse_q1, mae_q1 = model.evaluate(model_q1)
    results_q1.append([n, mse_q1, mae_q1])

    loss_dict_q1[str(n)+' samples'] = partial_loss_q1
    
    plt.ion()
    model.plot_curve(x_test_q1, y_true_q1, y_pred_q1,f"sample size = {n}")
    model.plot_loss_100(loss_q1, str(n)+' samples')
    plt.pause(0.01)
    
# 比较MSE/MAE/损失曲线
df_q1=pd.DataFrame(results_q1,columns=['Sample_Sizes','MSE','MAE'])
print(df_q1)

# model.plot_loss(loss_dict_q1, title='Loss Comparison (Different Sample Sizes)')




## ---------网络结构影响---------
# 网络结构
# hide_layers = [[20,20], [20,20,20], [50,50], [50,50,50],[50,50,50,50],[100,100]]
hide_layers = [[50,50,50]]

# 网络结构-MSE-MAE-训练时间对
results_q2 = []
# 网络结构-损失值对
loss_dict_q2 = {}
# 网络结构-训练时间对
time_list_q2 = []

for s in hide_layers:
    layers_q2 = [1] + s +[1]
    print('当前网络结构：',layers_q2)
    # 模型训练
    x_train_q2, y_train_q2, x_ref_q2, y_ref_q2 = model.generate_data()
    model_q2, loss_q2, partial_loss_q2, train_time_q2 = model.train_model(x_train_q2, y_train_q2, layers=layers_q2)

    # 模型评估
    x_test_q2, y_true_q2, y_pred_q2, mse_q2, mae_q2 = model.evaluate(model_q2)
    
    print('MSE:',mse_q2,'MAE:',mae_q2,'Train_time:',train_time_q2)

    results_q2.append([s, mse_q2, mae_q2, train_time_q2])
    loss_dict_q2[str(s)] = partial_loss_q2
    time_list_q2.append(train_time_q2)

    plt.ion()
    model.plot_curve(x_test_q2, y_true_q2, y_pred_q2, str(s))
    model.plot_loss_100(loss_q2, str(s))
    plt.pause(0.01)

# 比较MSE/MAE/损失曲线/训练时间
df_q2=pd.DataFrame(results_q2,columns=['Hide_Layers','MSE','MAE','Train_time'])
print(df_q2)

# model.plot_loss(loss_dict_q2, title='Loss Comparison (Different Network Structures)')

model.plot_train_time(hide_layers, time_list_q2, xlabel="Network Structure")




## ---------激活函数影响---------
# 激活函数
# activations = ['relu', 'tanh', 'sigmoid']
activations = ['tanh']

# 激活函数-MSE-MAE对
results_q3 = []
# 激活函数-损失值对
loss_dict_q3 = {}
# 激活函数-训练时间对
time_list_q3 = []

for act in activations:
    print('当前激活函数：',act)
    # 模型训练
    x_train_q3, y_train_q3, x_ref_q3, y_ref_q3 = model.generate_data()
    model_q3, loss_q3, partial_loss_q3, train_time_q3 = model.train_model(x_train_q3, y_train_q3, actn = act)

    #模型评估
    x_test_q3, y_true_q3, y_pred_q3, mse_q3, mae_q3 = model.evaluate(model_q3)

    print('Structure:', act, "MSE:", mse_q3, "MAE:", mae_q3, 'Train_Time:', train_time_q3)

    results_q3.append([act, mse_q3, mae_q3])
    loss_dict_q3[act] = partial_loss_q3
    time_list_q3.append(train_time_q3)

    plt.ion()
    model.plot_curve(x_test_q3, y_true_q3, y_pred_q3, act)
    model.plot_loss_100(loss_q3, act)
    plt.pause(0.01)

# 比较MSE/MAE/损失曲线/训练时间
df_q3=pd.DataFrame(results_q3,columns=['Activation','MSE','MAE'])
print(df_q3)

model.plot_train_time(activations, time_list_q3,xlabel="Activation Functions")

# model.plot_loss(loss_dict_q3, title='Loss Comparison (Different Activation Functions)')

plt.ioff()
plt.show()