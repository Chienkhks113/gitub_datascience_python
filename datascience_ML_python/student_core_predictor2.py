import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error,mean_squared_error , r2_score

# đổi sang mô hình này
from sklearn.svm import SVR

# mô hình 

from sklearn.linear_model import LinearRegression

# đổi mô hình 
from sklearn.ensemble import RandomForestRegressor

# đổi mô hình 
from sklearn.model_selection import GridSearchCV

# Đọc dữ liệu vào 
data = pd.read_csv("StudentScore.xls")
# profile = ProfileReport(data, title="Student score Report", explorative=True)
# profile.to_file("student_report.html")

# # Kiểm tra hệ số tương quan  vì hệ số tương quan cao thì ra dùng hệ số tuyển tính
# print(data[["math score" , "writing score" , "reading score"]].corr())


# Tách dữ liệu theo chiều ngang , dọc 
target = "math score"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

education_levels = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
]

genders = x_train["gender"].unique()
lunchs = x_train["lunch"].unique()
prep_courses = x_train["test preparation course"].unique()

#  cột điềm đọc
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_levels, genders, lunchs, prep_courses])),
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False)),
])

# tách dụng áp dụng transformer hay áp dụng cái gì lên đâu nhé
preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["reading score", "writing score"]),
    ("ord_feature", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nom_transformer, ["race/ethnicity"]), # code rate
])

# result = preprocessor.fit_transform(x_train)
# print 

reg = Pipeline(steps=[
    ("preprocessor ", preprocessor),
    ("regressor",RandomForestRegressor()
     # regress dùng tính trung bình
     # ramdom dùng class đa số
     
     )
])

params = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__criterion": ["squared_error", "absolute_error", "poisson"]
}

# Giusp để kiểm tra mô hình xem cái nào tốt hơn
model= GridSearchCV(
    estimator=reg,
    param_grid=params,
    scoring="r2", # thử tất cả 
    cv=6,
    verbose=2, # để số càng lớn in ra càng nhiều thứ
    n_jobs=6
)
# fit huấn luyện mô hình
model.fit(x_train, y_train)
# sau khi mô hình lưu trữ xong mặc định nó sẽ lưu trữ bộ dữ liệu tốt nhất

print(model.best_score_)
print(model.best_params_)
































# y_predict = reg.predict(x_test)
# #dự đoán , thực tế 
# # for pred , label in zip(y_predict,y_test):
# #     print("Prediction:{}.Actual value: {}".format(pred,label))

# # dự đoán thực tế metricx để đánh giá mô hình

# print("MAE: {}".format(mean_absolute_error(y_test,y_predict))) # 2 ông này càng bé càng tốt
# print("MSE: {}".format(mean_squared_error(y_test,y_predict))) # ông này càng bé càng tốt
# print("R2:{}".format(r2_score(y_test,y_predict))) # ông này càng lớn càng tốt , lost đánh giá một cái mô hình chính xác nhất 
# # ôn lại xem lại và kiểu tra xem các mô hình nó nằm ở phần nào trong skitlearn
# params = {
#     "n_estimators": [50, 100, 200],
#     "criterion": ["gini", "entropy", "log_loss"]
# }
# # Giusp để kiểm tra mô hình xem cái nào tốt hơn
# clf = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=100),
#     param_grid=params,
#     scoring="f1", # thử tất cả 
#     cv=6,
#     verbose=1, # để số càng lớn in ra càng nhiều thứ
#     n_jobs=6
# )
# # fit huấn luyện mô hình
# clf.fit(x_train, y_train)
# y_predict = clf.predict(x_test)
# print(clf.best_score_)
# print(clf.best_params_)
# print(classification_report(y_test, y_predict))










# # đánh giá mô hình dự vào MAE lỗi bậc một , MSE lỗi bậc hai , RMSE 3 cái này có đặc điểm nó càng bé thì càng tốt 
# # nhược điểm các dựa đoán bé bằng nào là tốt mỗi một bài toán khác nhau thì nó có các giá trị khác nhau
# # cố gắng tranh thủ trong hôm nay

#  # càng 0 càng tồi càng 1 càng tốt mô hình chụp ảnh á  hay mô hình metrics quantifying
