import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from Hom1.func_class import Hom1Model
model = Hom1Model()


## ----训练点数影响----
# 训练点数
sample_sizes = [15, 35, 55, 75, 100]

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

plt.ioff()     # 关闭交互模式
plt.show()     # 阻塞显示，窗口不会自动关闭
