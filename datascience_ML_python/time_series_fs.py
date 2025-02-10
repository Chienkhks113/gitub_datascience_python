import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
# dịch để cho mô hình dễ tiêu hóa bắt buộc phải cover về mô hình cho nó dễ tiêu hóa
def create_recursive_data(data , windows_size):
    i = 1 
    while i < windows_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i+=1 
    data["target"] = data["co2"].shift(-i)
    data = data.dropna(axis=0)
    return data
target = "time"
data = pd.read_csv("co2.csv")
# kiểm tra số ô bị khuyết 2284-2225
# print(data.info())
data["target"] = pd.to_datetime(data["time"])
# ở cuối ý bỏ đi thi ở dưới 
# cover cho toàn bộ dữ liệu 
# imputer = SimpleImputer(strategy="median")
# data["co2"] = imputer.fit_transform(data[["co2"]])
# bản chất nó là cùng một đối tượng lên ta dùng nội suy 
# đây chính là nội suy , dùng các điểm viền ở bên ngoài để dự đoán các giá trị ở giữa nhé
data["co2"] = data["co2"].interpolate()
# bản chất chúng là cùng 1 đối tượng 
# vẽ khung con
# fig , ax = plt.subplots()
# ax.plot(data["time"],data["co2"])
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# plt.show()
windows_size = 5
data = create_recursive_data(data,windows_size)

# chia dữ liệu 
x = data.drop([target , "target"], axis = 1)
y = data[target]
# Vì sao ở đây ta chia train _slit ? nó chia ngẫu nhiên , lên ta chia theo kiểu kia cho nó hợp lệ

# chia 25% cho train , 25% cho test
train_ratio = 0.8
num_samples = len(x)
x_train = x[:int(num_samples*train_ratio)]
y_train = y[:int(num_samples*train_ratio)]
y_test = x[int(num_samples*train_ratio):]
y_test = y[:int(num_samples*train_ratio):]