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
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV

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



# Yêu cầu thay đổi thử mô hình xem cái này tốt hơn
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
    "regressor__criterion": ["squared_error", "absolute_error", "poisson"],
    # thử xem cái mô hình nào tốt hơn
    "regressor__max_depth":[None,2,5],
    "regressor__min_samples_split":[2,5,10],
    "regressor__min_samples_leaf":[1,2,5]

    # thử xem đang sai dòng này
    # "preprocessor__num_feature__imputer__strategy":["mean","median"]
}

# Giusp để kiểm tra mô hình xem cái nào tốt hơn
model= RandomizedSearchCV(
    estimator=reg,
    param_distributions=params,
    # tôi chỉ thử ngẫu nhiên 30 tổ hợp thôi
    n_iter=30,
    scoring="r2", # thử tất cả 
    cv=6,
    verbose=1, # để số càng lớn in ra càng nhiều thứ
    n_jobs=6
)
# fit huấn luyện mô hình
model.fit(x_train, y_train)
# sau khi mô hình lưu trữ xong mặc định nó sẽ lưu trữ bộ dữ liệu tốt nhất

print(model.best_score_)
print(model.best_params_)

# sau đó lại đem mô hình đi dự đoán



y_predict = model.predict(x_test)
#dự đoán , thực tế 
# for pred , label in zip(y_predict,y_test):
#     print("Prediction:{}.Actual value: {}".format(pred,label))

# # dự đoán thực tế

# print("MAE: {}".format(mean_absolute_error(y_test,y_predict))) # 2 ông này càng bé càng tốt
# print("MSE: {}".format(mean_squared_error(y_test,y_predict))) # ông này càng bé càng tốt
# print("R2:{}".format(r2_score(y_test,y_predict))) # ông này càng lớn càng tốt , lost đánh giá một cái mô hình chính xác nhất 



#hyper parameters : là tham số đối với toàn bộ dữ liệu
#parameters : nó là tham số đối với mô hình nào 



# 0.8375388346392644
# {'regressor__n_estimators': 100 , số lượng cây tốt nhất là 100
# , 'regressor__min_samples_split': 10,
#  'regressor__min_samples_leaf': 1, , lá tốt nhất là một
# 'regressor__max_depth': None, 'regressor__criterion': 'squared_error'}








