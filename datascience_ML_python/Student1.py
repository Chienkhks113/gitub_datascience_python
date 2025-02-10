import pandas as pd
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics  import accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

data = pd.read_csv("StudentScore.xls")
# profile = ProfileReport(data, title="Student_Score", explorative=True)
# profile.to_file("student_report.html")

# cover  thử hệ số tương quan vì thế ta lên chọn hệ số tuyến tính
print(data[["math score" , "writing score" , "reading score"]].corr())


target = "math score"
x = data.drop(target, axis = 1)
# axis loại bỏ theo cột
y=data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=5)
# print(data["parental level of education"].unique())

# code này dùng để mãu hóa
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),  # Thực viện này dùng để điền vào các giá trị thiếu
    ("scaler", StandardScaler())
])
result3 = num_transformer.fit_transform(x_train[["reading score" , "writing score"]])
# # result1 = num_transformer.transform(x_train[["reading score"]])
print(result3)


# Bắt buộc phải chỉ ra các cột
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
#  mã hóa ordinol code của cái đó nhé
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_levels, genders, lunchs, prep_courses])),
])
result = ord_transformer.fit_transform(x_train[["reading score" , "writing score"]])
# result1 = num_transformer.transform(x_train[["reading score"]])
print(result)

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False)),
])

result2 = nom_transformer.fit_transform(x_train[["reading score" , "writing score"]])
# result1 = num_transformer.transform(x_train[["reading score"]])
print(result2)












# # tách dụng áp dụng transformer hay áp dụng cái gì lên đâu nhé
# preprocessor = ColumnTransformer(transformers=[
#     ("num_feature", num_transformer, ["reading score", "writing score"]),
#     ("ord_feature", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
#     ("nom_feature", nom_transformer, ["race/ethnicity"]), # code rate
# ])

# # result = preprocessor.fit_transform(x_train)
# # print 

# reg = Pipeline(steps=[
#     ("preprocessor . preprocessor"),
#     ("regressor",LinearRegression())
# ])
# reg.fit(x_train,y_train)
# y_predict = reg.predict(x_test)
# #dự đoán , thực tế 
# # for pred , label in zip(y_predict,y_test):
# #     print("Prediction:{}.Actual value: {}".format(pred,label))

# # dự đoán thực tế

# print("MAE: {}".format(mean_absolute_error(y_test,y_predict)))
# print("MSE: {}".format(mean_squared_error(y_test,y_predict)))
# print("R2:{}".format(r2_score(y_test,y_predict)))















# đánh giá mô hình dự vào MAE lỗi bậc một , MSE lỗi bậc hai , RMSE 3 cái này có đặc điểm nó càng bé thì càng tốt 
# nhược điểm các dựa đoán bé bằng nào là tốt mỗi một bài toán khác nhau thì nó có các giá trị khác nhau
# cố gắng tranh thủ trong hôm nay

 # càng 0 càng tồi càng 1 càng tốt mô hình chụp ảnh á  hay mô hình metrics quantifying



