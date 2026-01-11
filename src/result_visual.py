import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. 加载结果
y_test = np.load("../results/y_test.npy")
y_pred = np.load("../results/y_pred.npy")
history = np.load("../results/history.npy", allow_pickle=True).item()
params = np.load("../results/scaler_params.npy")

# 2. 手动恢复归一化器进行反转
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = params[0], params[1]

y_true_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))

# 3. 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("../results/loss_curve.png")

# 4. 绘制负荷预测对比图 (取最近一周即168小时数据)
plt.figure(figsize=(15, 6))
plt.plot(y_true_inv[-168:], label='Actual Load', color='blue')
plt.plot(y_pred_inv[-168:], label='Predicted Load', color='red', linestyle='--')
plt.title('Power Load Prediction (Last 168 Hours)')
plt.ylabel('Load (kW)')
plt.legend()
plt.savefig("../results/prediction_contrast.png")
plt.show()