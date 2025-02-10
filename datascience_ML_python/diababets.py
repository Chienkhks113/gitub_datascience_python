import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics  import accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


from sklearn.linear_model import LogisticRegression

data = pd.read_csv("diabetes.csv")
# Dùng để đọc dữ liệu thống kê dữ liệu
# profile = ProfileReport(data, title="Diabetes Report", explorative=True)
# profile.to_file("diabetes_report.html")
# Hệ số tương quan thấp dùng phi tuyến
#Loại bỏ xem lại video chỗ nào
# Phân chia dữ liệu theo chiều dọc để đẩy future sang 1 bên target sang 1 bên
# chia ta chiều dọc tách x sang 1 bên y sang 1 bên
target = "Outcome"
x = data.drop("Outcome", axis = 1)
# axis loại bỏ theo cột
y=data[target]

# Bộ test để kiểm tra không bảo giờ đc liên quan đến quá trình huấn luyện chỉ fit train thôi nhé , chú ý nhé
# ý dữ liệu thì phần trăm lớn hơn cho bộ train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=5)
#random_state cố định số này thì ngẫm nhiên nó sẽ cố định
print(x_train.shape , y_train.shape) 
print(x_test.shape , y_test.shape)

# Bộ test khác nhau k thể so sánh được mô hình , các mô hình chỉ so sánh khi nó cùng bộ test
# Bộ validation
# vì sao ta chia 60 , 20 ,20 thì ta lại ghi là 0.25 =20/80
# 
# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25 , random_state=5)
# print(X_train)
# print(x_val.shape)

# Tiền xử lý 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # tính ra cách giải giá trị cùng giải giá trị
print(x_train)
# biến đổi bộ k đc sử dụng fit_transform cho bộ test nhé , với từng cột nó sẽ tính kì vọng theo bộ train thôi nhé
x_test = scaler.transform(x_test)
# print(scaler.fit(data)) # đo xem chỉ số đo như nào bước này tính đo đạc  tính kì vọng và độ lệch chuẩn
# print(scaler.mean_)
# print(scaler.transform(data)) # ông ấy cắt so đo theo cái mảng vải , biến đổi theo số đo mà ông ấy vừa
# print(scaler.transform([[2, 2]])) # đo đạc , biến đổi


# Huấn luyện mô hình 
clt = LogisticRegression()
# mô hình randomForest clf = RandomForestClassifier()
clt.fit(x_train,y_train)
# cái chỗ này điểm khác 



# Xem lại cái này nhé
y_predict = clt.predict_proba(x_test)

# Tìm sẽ tìm ra ngưỡng
print(y_predict)
# Bài toán này chỉ có 0 với 1 thôi