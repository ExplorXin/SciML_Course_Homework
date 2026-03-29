import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from Hom1.func_class import Hom1Model
model = Hom1Model()


## ----激活函数影响----
# 激活函数
activations = ['relu', 'tanh', 'sigmoid']

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
    plt.pause(0.01)

# 比较MSE/MAE/损失曲线/训练时间
df_q3=pd.DataFrame(results_q3,columns=['Activation','MSE','MAE'])
print(df_q3)

model.plot_train_time(activations, time_list_q3,xlabel="Activation Functions")

model.plot_loss(loss_dict_q3, title='Loss Comparison (Different Activation Functions)')

plt.ioff()     # 关闭交互模式
plt.show()     # 阻塞显示，窗口不会自动关闭