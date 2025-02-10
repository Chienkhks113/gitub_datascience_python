import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error,mean_squared_error , r2_score


# mô hình 

from sklearn.linear_model import LinearRegression

# Đọc dữ liệu vào 
data = pd.read_csv("StudentScore.xls")
# profile = ProfileReport(data, title="Student score Report", explorative=True)
# profile.to_file("student_report.html")
# # Kiểm tra hệ số tương quan  vì hệ số tương quan cao thì ra dùng hệ số tuyển tính
# print(data[["math score" , "writing score" , "reading score"]].corr()) //  chúng ta chỉ xét các cột này thôi nhé có các cột không phải là số lên chính vì thế ta phải làm như thế
'''
Nhìn vào hệ số tương quan các cái feature hệ số tương quan cao lên chọn mô hình tuyến tính 
'''
# Tách dữ liệu theo chiều ngang , dọc 
target = "math score"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  #Điền vào giá trị bị thiếu median:trung vị
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

#  cột điền đọc
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")), # Điền vào giá trị bị thiếu
    ("encoder", OrdinalEncoder(categories=[education_levels, genders, lunchs, prep_courses])),
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")), #Điền vào giá trị bị thiếu
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

'''
Phân biệt các cột : 
parental level of education:ordinal
associate: liên thông 

'''
reg = Pipeline(steps=[
    ("preprocessor . preprocessor"),
    ("regressor",LinearRegression())
])
reg.fit(x_train,y_train)
y_predict = reg.predict(x_test)
#dự đoán , thực tế 
# for pred , label in zip(y_predict,y_test):
#     print("Prediction:{}.Actual value: {}".format(pred,label))
# dự đoán thực tế
print("MAE: {}".format(mean_absolute_error(y_test,y_predict)))
print("MSE: {}".format(mean_squared_error(y_test,y_predict)))
print("R2:{}".format(r2_score(y_test,y_predict)))


# đánh giá mô hình dự vào MAE lỗi bậc một , MSE lỗi bậc hai , RMSE 3 cái này có đặc điểm nó càng bé thì càng tốt 
# nhược điểm các dựa đoán bé bằng nào là tốt mỗi một bài toán khác nhau thì nó có các giá trị khác nhau
# cố gắng tranh thủ trong hôm nay

 # càng 0 càng tồi càng 1 càng tốt mô hình chụp ảnh á  hay mô hình metrics quantifying
