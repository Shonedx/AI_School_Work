import numpy as np
import os
# 导入模型定义函数
from model_build import build_lstm_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# 1. 加载数据
# 确保路径指向 results 文件夹
X_train = np.load("../results/X_train.npy")
X_test = np.load("../results/X_test.npy")
y_train = np.load("../results/y_train.npy")
y_test = np.load("../results/y_test.npy")

# 2. 初始化模型
# X_train.shape[1] 是时间步长（24），1 是特征数
input_shape = (X_train.shape[1], 1)
model = build_lstm_model(input_shape)

# 3. 开始训练
print("开始模型训练，请耐心等待...")
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=30,
    validation_data=(X_test, y_test),
    verbose=1
)

# 4. 保存训练产物（这是可视化脚本运行的前提！）
print("正在保存训练结果...")
model.save("../results/lstm_model.h5") # 保存模型文件
np.save("../results/history.npy", history.history) # 保存损失曲线数据

# 5. 进行预测
y_pred = model.predict(X_test)
np.save("../results/y_pred.npy", y_pred) # 保存预测值

# 6. 计算评估指标并打印 (实训报告必填)
# 注意：这里计算的是归一化后的指标，仅供训练参考
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*30)
print("模型训练并保存完成！")
print(f"测试集 MAE (归一化): {mae:.4f}")
print(f"测试集 RMSE (归一化): {rmse:.4f}")
print("="*30)