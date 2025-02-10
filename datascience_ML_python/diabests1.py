import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("diabetes.csv")
# profile = ProfileReport(data, title="Diabetes Report", explorative=True)
# profile.to_file("diabetes_report.html")
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

params = {
    "n_estimators": [50, 100, 200],   # Tối ưu cái k-flod cho nó, vì thế dùng kĩ thuật đó , lưu ý mặc định valdian nó sẽ giữ nguyên trong bộ
    "criterion": ["gini", "entropy", "log_loss"]
}

# Giusp để kiểm tra mô hình xem cái nào tốt hơn

# Cái này dùng kĩ thuật k-klod kĩ thuật tối ưu bộ valdiation để làm giảm
clf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=100),
    param_grid=params,
    scoring="f1", # thử tất cả 
    cv=6,
    verbose=1, # để số càng lớn in ra càng nhiều thứ
    n_jobs=6
)
# fit huấn luyện mô hình
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(clf.best_score_)
print(clf.best_params_)
print(classification_report(y_test, y_predict))





# Với bài toán classification mặc định là accuracy





 #Weighted avg trung bình có trọng số

 # Bài tập xử lý nhé

#  k_fold cross vadidation hay kiểm tra chéo
'''
1. Xác định các tham số cho mô hình
params = { "n_estimators": [50, 100, 200],
           "criterion": ["gini", "entropy", "log_loss"]
}

params: Đây là một từ điển chứa các tham số sẽ được tối ưu
n_estimators : So lượng cây trong rừng (Random Forest). Các giá trị
thử nghiệm là 50 , 100 , 200
criterion : Tiêu chí để đánh giá chất lượng của một tách. Các lựa chọn là 
"gini" , "entropy" và "log_loss"

2. Thiết lập GridSearchCV để tìm kiếm các tham số tốt nhất
clf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=100),
    param_grid=params,
    scoring="f1",  # thử tất cả 
    cv=6,
    verbose=1,  # để số càng lớn in ra càng nhiều thứ
    n_jobs=6
)
clf: Đây là đối tượng GridSearchCV sẽ thực hiện tìm kiếm lưới (grid search) để tìm các tham số tốt nhất cho mô hình RandomForestClassifier.

estimator: Mô hình cơ sở để tối ưu hóa, ở đây là RandomForestClassifier với random_state=100.

param_grid: Lưới các tham số cần tối ưu hóa, ở đây là params.

scoring: Chỉ số để đánh giá mô hình, ở đây là f1 (F1 score).

cv: Số lượng gấp chéo (cross-validation folds), ở đây là 6.

verbose: Mức độ chi tiết của thông tin đầu ra. Giá trị càng lớn thì in ra càng nhiều thông tin.

n_jobs: Số lượng công việc chạy song song, ở đây là 6.

'''