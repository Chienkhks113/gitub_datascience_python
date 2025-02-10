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
# result3 = num_transformer.fit_transform(x_train[["reading score" , "writing score"]])
# # # result1 = num_transformer.transform(x_train[["reading score"]])
# print(result3)


# Bắt buộc phải chỉ ra các cột
education_levels = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
]
genders =["female","male"]
genders = x_train["gender"].unique()
lunchs = x_train["lunch"].unique()
prep_courses = x_train["test preparation course"].unique()
#  mã hóa ordinol code của cái đó nhé
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_levels, genders])),
])
# result = ord_transformer.SSfit_transform(x_train[["reading score" , "writing score"]])
# # result1 = num_transformer.transform(x_train[["reading score"]])
# print(result)

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False)),
])

# result2 = nom_transformer.fit_transform(x_train[["reading score" , "writing score"]])
# # result1 = num_transformer.transform(x_train[["reading score"]])
# print(result2)




# 2 ông dùng để k có thứ bậc
# từ 3 trở lên mới có thứ tự




