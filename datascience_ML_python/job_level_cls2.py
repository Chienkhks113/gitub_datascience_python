import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import re
#Hàm giúp chúng ta lọc trpng onehot
def filter_location(loc):
    result = re.findall(r",\s[A-Z]{2}", loc)
    if len(result):
        return result[0][2:]
    else:
        return loc


data = pd.read_excel("final_project.ods" , engine="odf" , dtype=str)

data["location"] = data["location"].apply(filter_location)
data = data.dropna()



# Xử lý dữ liệu loại bỏ các hàng trống
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


# Kiểm tra dữ liệu xem nó ở dạnh nào
result = model.fit(x_train , y_train)
print(result.shape)




# Cần fix chỗ này 