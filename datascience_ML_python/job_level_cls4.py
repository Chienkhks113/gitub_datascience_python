import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from  sklearn.metrics import classification_report
import re

#Hàm giúp chúng ta lọc trpng onehot
def filter_location(loc):
    result = re.findall(r",\s[A-Z]{2}", loc)
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
    ("des_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), "description"),
    # ("function_feature", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry")
])


model = Pipeline(steps=[
    ("preprocessor ", preprocessor), # 85k features
    ("regressor",RandomForestClassifier())
])
# result = model.fit_transform(x_train)
# print(result)


# Kiểm tra dữ liệu xem nó ở dạnh nào vì sao x_train, y_train cho vào nhé ??
model.fit(x_train , y_train)
y_predict = model.predict(x_test)
print(classification_report(y_test,y_predict))

# max df , min df dùng để thu nhỏ lại tầng số của nó đi 


# Cần fix chỗ này 



#   precision    recall  f1-score   support

#                         bereichsleiter       0.75      0.05      0.09       192
#          director_business_unit_leader       1.00      0.29      0.44        14

#                               accuracy                           0.69      1615 
#                              macro avg       0.51      0.30      0.32      1615 
#                           weighted avg       0.69      0.69      0.64      1615 