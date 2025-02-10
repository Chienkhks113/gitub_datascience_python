import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from  sklearn.metrics import classification_report
import re
from sklearn.feature_selection import SelectKBest

def filter_location(loc):
    result =re.findall("\,\s[A-Z]{2}",loc)
    if len(result):
        return result[0][2:]
    else:
        return loc
data = pd.read_excel("final_project.ods" , engine="odf" , dtype=str)
data = data.dropna()
# xử lý code dicription  xử kiểm tra và xử lý khi nó bị lan 
'''
data = data.dropna()
print(data.isna().sum()) '''


# kiểm tra xem nó có mấy sample
# print(data["career_level"].value_counts())
#Tách theo chiều dọc , tách theo chiều ngang  x sang một bên y sang một bên


data["location"] = data["location"].apply(filter_location)


target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
# Tách theo chiều ngang
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2,random_state=42,stratify=y)

# Tiền sử lý dữ liệu

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("nom_feature",OneHotEncoder(handle_unknown="ignore"),["location","function"]),
    # ("location_feature", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("des_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 2) , min_df=0.01 ,max_df=0.95), "description"),# Câu lệnh làm giảm số lượng performance
    # ("function_feature", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry")
])


model = Pipeline(steps=[
    ("preprocessor ", preprocessor), # 7976 features
    # giữ lại feature nào có ảnh hưởng nhất
    # giảm số lượng feature không cần thiết ===> giảm rồi sau đó bắt buộc phải kiểm tra lại xem ảnh hưởng đến performment không nhé 
    ("feature_selector",SelectKBest(score_func=chi2,k=800)), # sửa lỗi code chỗ này 
    ("regressor",RandomForestClassifier(random_state=42))

     
])
result = model.fit_transform(x_train)
print(result.shape)

# Kiểm tra dữ liệu xem nó ở dạnh nào vì sao x_train, y_train cho vào nhé ??
# model.fit(x_train , y_train)
# y_predict = model.predict(x_test)
# print(classification_report(y_test,y_predict))