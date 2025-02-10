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
# Dùng hàm cho nó
data["location"] = data["location"].apply(filter_location)
# Xử lý dữ liệu bị khuyết
data = data.dropna()
# xử lý code dicription  xử kiểm tra và xử lý khi nó bị lan 
'''
data = data.dropna()
print(data.isna().sum()) '''

# kiểm tra xem nó có mấy sample
# print(data["career_level"].value_counts())
#Tách theo chiều dọc , tách theo chiều ngang  x sang một bên y sang một bên
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
# Tách theo chiều ngang
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2,random_state=42,stratify=y)



# Tiền xử lý dữ liệu
'''
Áp dụng bộ chuyển đổi cho các cột của mảng và pandas DataFrame 
Bộ ước tính này cho phép các cột hoặc các tập tập hợp khác nhau của đầu vào 
được chuyển đổi riêng biệt và các tính năng được tạo ra bởi mỗi bộ chuyển sẽ được
nối lại để tạo thành một không gian tính năng duy nhất. Điều này hữu ích cho dữ liệu không
đồng nhất hoặc dữ liệu dạng cột , để kết hợp một số cơ chế trích xuất năng hoặc chuyển đổi thành một 
bộ chuyển duy nhất
'''

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("nom_feature",OneHotEncoder(),["location","function"]),
    # ("location_feature", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("des_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), "description"),
    # ("function_feature", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry")
])


model = Pipeline(steps=[
    ("preprocessor ", preprocessor), # 85k features
     ("regressor",RandomForestClassifier())

     
])
result = model.fit_transform(x_train)
print(result)


# Kiểm tra dữ liệu xem nó ở dạnh nào
# model.fit(x_train , y_train)
# y_predict = model.predict(x_test)
# print(classification_report(y_test,y_predict))




# Cần fix chỗ này 