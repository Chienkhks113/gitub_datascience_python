import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_direct_data(data, windows_size, target_size):
    i = 1
    while i < windows_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    i = 0
    while i < target_size:
        data["target_{}".format(i)] = data["co2"].shift(-i-windows_size)
        i += 1
    data = data.dropna(axis=0)
    return data


data = pd.read_csv("co2.csv")
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()

# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# plt.show()

windows_size = 5
target_size = 3
data = create_direct_data(data, windows_size, target_size)
targets = ["target_{}".format(i) for i in range(target_size)]
x = data.drop(["time"] + targets, axis=1)
y = data[targets]

train_ratio = 0.8
num_samples = len(x)

x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]
r2 = []
mse = []
mae = []
regs = [LinearRegression() for _ in range(target_size)]
for i, reg in enumerate(regs):
    reg.fit(x_train, y_train["target_{}".format(i)])
    y_predict = reg.predict(x_test)
    r2.append(r2_score(y_test["target_{}".format(i)], y_predict))
    mse.append(mean_squared_error(y_test["target_{}".format(i)], y_predict))
    mae.append(mean_absolute_error(y_test["target_{}".format(i)], y_predict))


print("R2 score: {}".format(r2))
print("MSE: {}".format(mse))
print("MAE: {}".format(mae))





# chú nhiều mô hình dự đoán trong thực tế , mỗi 1 mô hình sẽ chịu trách nhiệm dự đoán trong tương lại 
#ưu điểm : luôn có đúng và nhiều mô hình nhiều bộ nhớ 




