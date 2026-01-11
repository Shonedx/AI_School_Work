import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# 创建结果保存目录
os.makedirs("../results", exist_ok=True)

# 1. 读取数据
# 针对你的CSV格式：,数据时间,总有功功率（kw）
file_path = "../data/power_load.csv"
if not os.path.exists(file_path):
    print(f"错误：未找到文件 {file_path}，请确认文件名和路径！")
    exit()

data = pd.read_csv(file_path)

# 2. 严格按位置提取数据并修复对齐问题
# iloc[:, 1] 是“数据时间”，iloc[:, 2] 是“总有功功率（kw）”
times = pd.to_datetime(data.iloc[:, 1])
load_values = pd.to_numeric(data.iloc[:, 2], errors='coerce').values # 使用.values防止索引对齐失败

# 3. 重新构建干净的 DataFrame
clean_df = pd.DataFrame({'负荷值': load_values}, index=times)
print(f"原始数据点数: {len(clean_df)}")

# 4. 数据重采样：将15分钟数据转化为小时平均值
data_hourly = clean_df.resample('H').mean()

# 5. 缺失值填充：先插值，再用前后值填充边缘
data_hourly["负荷值"] = data_hourly["负荷值"].interpolate(method='linear')
data_hourly["负荷值"] = data_hourly["负荷值"].ffill().bfill()

# 6. 异常值处理：3σ原则剔除
mean = data_hourly["负荷值"].mean()
std = data_hourly["负荷值"].std()
data_hourly = data_hourly[(data_hourly["负荷值"] >= mean - 3*std) & (data_hourly["负荷值"] <= mean + 3*std)]

# 7. 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_hourly["负荷值"].values.reshape(-1, 1))

# 8. 构建时序序列：用前24小时预测下1小时
time_step = 24
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X).reshape(-1, time_step, 1)
y = np.array(y)

# 9. 划分并保存
train_size = int(0.8 * len(X))
np.save("../results/X_train.npy", X[:train_size])
np.save("../results/X_test.npy", X[train_size:])
np.save("../results/y_train.npy", y[:train_size])
np.save("../results/y_test.npy", y[train_size:])
# 保存 scaler 的参数以便后续反归一化
np.save("../results/scaler_params.npy", [scaler.min_, scaler.scale_])

print("数据预处理成功完成！")
print(f"最终用于训练的小时级样本数: {len(X)}")