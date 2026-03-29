import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from Hom1.func_class import Hom1Model
model = Hom1Model()


## ----网络结构影响----
# 网络结构
hide_layers = [[20,20], [20,20,20], [50,50], [50,50,50],
               [50,50,50,50],[100,100]]

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
    plt.pause(0.01)

# 比较MSE/MAE/损失曲线/训练时间
df_q2=pd.DataFrame(results_q2,columns=['Hide_Layers','MSE','MAE','Train_time'])
print(df_q2)


model.plot_loss(loss_dict_q2, title='Loss Comparison (Different Network Structures)')

model.plot_train_time(hide_layers, time_list_q2, xlabel="Network Structure")

plt.ioff()     # 关闭交互模式
plt.show()     # 阻塞显示，窗口不会自动关闭