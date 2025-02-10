import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Hàm tạo dữ liệu đệ quy
def create_recursive_data(data, window_size):
    for i in range(1, window_size + 1):
        data["co2_{}".format(i)] = data["co2"].shift(-i)
    data["target"] = data["co2"].shift(-window_size)
    data = data.dropna(axis=0)
    return data

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv("co2.csv")

# Kiểm tra số ô bị khuyết
print(data.info())

# Chuyển đổi cột thời gian
data["time"] = pd.to_datetime(data["time"])

# Điền dữ liệu bị khuyết bằng phương pháp nội suy
data["co2"] = data["co2"].interpolate()

# Cấu hình window_size
window_size = 5
data = create_recursive_data(data, window_size)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x = data.drop(["time", "target"], axis=1)
y = data["target"]

# Tỷ lệ chia tập huấn luyện và kiểm tra
train_ratio = 0.8
num_samples = len(x)
train_size = int(num_samples * train_ratio)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Huấn luyện mô hình hồi quy tuyến tính
reg = LinearRegression()
reg.fit(x_train, y_train)

# Dự đoán trên tập kiểm tra
y_predict = reg.predict(x_test)

# Đánh giá mô hình
print("R2 score: {}".format(r2_score(y_test, y_predict)))
print("MSE score: {}".format(mean_squared_error(y_test, y_predict)))
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))





 # xem thêm video này  ML/DL/CV Nâng cao phút thứ 43:34