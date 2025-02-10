import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def create_recursive_data(data, windows_size):
    i = 1
    while i < windows_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data["target"] = data["co2"].shift(-i)
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
data = create_recursive_data(data, windows_size)
x = data.drop(["time", "target"], axis=1)
y = data["target"]
train_ratio = 0.8
num_samples = len(x)
x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
# for i, j in zip(y_predict, y_test):
#     print("Prediction: {}. Actual value: {}".format(i, j))
print("R2 score: {}".format(r2_score(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))

fig, ax = plt.subplots()
ax.plot(data["time"][:int(num_samples * train_ratio)], data["co2"][:int(num_samples * train_ratio)], label="train")
ax.plot(data["time"][int(num_samples * train_ratio):], data["co2"][int(num_samples * train_ratio):], label="test")
ax.plot(data["time"][int(num_samples * train_ratio):], y_predict, label="prediction")
ax.set_xlabel("Time")
ax.set_ylabel("CO2")
ax.legend()
plt.show()

# current_data = [380.5, 390, 390.2, 390.4, 394.2]
# dự đoán cho 10 tuần khác nhau 
# for i in range(10):
#     prediction = reg.predict([current_data]).tolist()
#     print("Input is {}. CO2 in week {} is {}".format(current_data, i, prediction[0]))
#     current_data = current_data[1:] + prediction





#  MÔ HÌNH NÀY : chúng ta tìm hiểm 2 các tiếp cận : dùng một mô hình chỉ có 1 mô hình càng giá trị về sao giá trị càng bị sai

# có 2 cách tiếp cận recursive_time_series and direct_time_fs




#  Chú ý cái này nhé regularization :  l1 ép cho cái thằng này 1 hoặc một vài cái đó về không 
                                #    l2 ép gần k thôi