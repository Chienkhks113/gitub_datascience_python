import pandas as pd  
# import thư viện pandas để theo tác dữ liệu (đọc file CSV , tạo DataFrame)
from ydata_profiling import ProfileReport # từ thư viện ydata_profiling để phân tích dữ liệu khám phá
from sklearn.model_selection import train_test_split
# import hàm tran_test_split từ thư viện scikit learn để chia dữ liệu thành các tập hợp huấn luyện và kiểm thử
from sklearn.preprocessing import StandardScaler
# import lớm StandardScaler từ scikit-learn để chia dữ liệu thành các tập hợp huấn luyện và kiểm thử
from sklearn.svm import SVC
#import lớp StandardScaler tử scokit-learn để tiền xử lý dữ liệu (chuẩn hóa các đặc trưng)
from sklearn.linear_model import LogisticRegression, LinearRegression
#import linearRegression từ scokit-learn để xây dựng mô hình hồi quy tuyến tính
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
# import lớp RandomForestClassifier từ scikit-learn để xây dựng mô hình phân loại rừng ngẫu nhiên 
from sklearn.model_selection import GridSearchCV
#import lớp GridSearchCv TỪ SCOKIT-LEARN để điều chỉnh siêu tham số
from lazypredict.Supervised import LazyClassifier
#import lớp LazyClassifier từ thư viện lazypredict để học máy tự động
from sklearn.metrics import mean_squared_error
#: Import hàm mean_squared_error từ scikit-learn để đánh giá hiệu suất của mô hình (không được sử dụng trong đoạn mã cụ thể này).
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Đọc dữ liệu từ file" diabetes.csv" và lưu vào một DataFrame của Pandas 
data = pd.read_csv("diabetes.csv")
'''
# profile = ProfileReport(data, title="Diabetes Report", explorative=True) (bị chú thích): Tạo một đối tượng ProfileReport để phân tích dữ liệu khám phá,
#  nhưng nó bị chú thích trong đoạn mã này.
#  Bỏ chú thích sẽ tạo ra một báo cáo có tên "diabetes_report.html" 
# (giả sử ydata_profiling được cài đặt).
'''
# profile = ProfileReport(data, title="Diabetes Report", explorative=True)
# profile.to_file("diabetes_report.html")
# Phân chia dữ liệu theo chiều dọc
'''
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,
DiabetesPedigreeFunction,Age,Outcome '''
'''
target="Outcome":Xác định biến đích(biến phụ thuộc) là cột có tên "Outcome" trong DataFrame
x=data.drop(target,axis=1):Tạo một DataFrame mới tên x chứa tất cả các cột ngoại từ"OutCome"(các biến độc lập)
y=data[target]:Tạo Pandas Series tên y chỉ chứa cột"Outcome"
'''
target = "Outcome"
# phân chia theo chiều dọc
x = data.drop(target, axis=1)
y = data[target]
# Phân chia dữ liệu theo chiều ngang
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
'''
Chia tập dữ liệu huấn luyện và kiểm thử:
x_train,x_test,y_train,y_test = train_test_split(x,t , test_size=0.2, random_state=42)
Chia dữ liệu thành các tập huấn luyện và kiểm thử
test_size=0.2
chỉ định bằng 20% dữ liệu sẽ sử dụng để kiểm thử
random_state=42 đảm bảo khả năng tái tạo bằng cách đặt seed cho trình tạp ngẫu nhiên
'''


'''
Chuẩn hóa Đặc Trưng 
scaler = StandardScaler():Tạo một đối tượng StandardScaler để chuẩn hóa các đặc trung trong x_train
x_train = scaler.fit_transform(x_train) áp dựng scaler vào dữ lueeju huấn luyện x_train
và chuyển đối nó bằng cách đưa trung bình của mỗi đặc trưng về 0 và độ lệch chuẩn về 1 
Điều này quan trong cho hồi quy tuyến tính để tránh các đặc trung có thanh đó lớn hớn 
chi phối mô hình 
x_test = scaler.transform(x_test): 
chuyển đổi dữ liệu kiểm thử x_test bằng các sử dụng cùng một scaler
để áp dựng cho dữ lueeju huấn luyện và điều này đảm bảo rằng dữ liệu kiểm thử
có cùng thang đó với dữ liệu huấn luyện
'''
# Tiền xử lý dữ liệu 
scaler = StandardScaler()
# fit đánh giá tính +biến đổi , chỉ xử dụng x_train
x_train = scaler.fit_transform(x_train)
# transform biến đổi dựa vào  fit luôn
x_test = scaler.transform(x_test)
clf = LazyClassifier(verbose=2, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
# k-fold kiểu tra chéo để tách train và val 
# Sau đó dùng cái này sẽ kiểm tra được cái mô hình nào có thể gần các giá trị đúng với cái đó nhất

# bộ hyermeter tối ưu , ở đây ta chia thafh 6 mản khác nhau
params = {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"]
    #Phần này lên đọc tài liệu và kiểm tra code
}

# tìm bộ tham số tối ưu của cái bộ này
clf = GridSearchCV(
    # chỉ sử dụng cho mô hình này để hoàn thành bộ dữ liệu này 
    estimator=RandomForestClassifier(random_state=100),
    param_grid=params,
    scoring="f1",
    cv=6,
    verbose=1,
    n_jobs=6
)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(clf.best_score_)
print(clf.best_params_)
print(classification_report(y_test, y_predict))

# Fitting 6 folds for each of 9 candidates, totalling 54 fits
# 0.6492307067754637
# {'criterion': 'gini', 'n_estimators': 200}
#               precision    recall  f1-score   support

#            0       0.80      0.78      0.79        99
#            1       0.62      0.65      0.64        55

#     accuracy                           0.73       154
#    macro avg       0.71      0.72      0.71       154
# weighted avg       0.74      0.73      0.74       154