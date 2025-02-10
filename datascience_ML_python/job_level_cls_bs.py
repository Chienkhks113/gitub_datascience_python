import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import re

#Hàm giúp chúng ta lọc trpng onehot
def filter_location(loc):
    result = re.findall(r",\s[A-Z]{2}", loc)
    if len(result):
        return result[0][2:]
    else:
        return loc

data = pd.read_excel("final_project.ods" , engine="odf" , dtype=str)
# với mỗi 1 giá trị của cái data ta sẽ apply cái hàm filer_location lại với nhau
data["location"] = data["location"].apply(filter_location)
print(data["career_level"].value_counts())

# xử lý code dicription  xử kiểm tra và xử lý khi nó bị khuyết đi một hàng, ta phải xử lý cách là vứt nó đi
data = data.dropna()
print(data.isna().sum()) 

# ho vào applice vào cái hàng đos để sử dụng cho one hot

# kiểm tra xem nó có mấy sample
# print(data["career_level"].value_counts())
#Tách theo chiều dọc , tách theo chiều ngang  x sang một bên y sang một bên
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
# Tách theo chiều ngang
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2,random_state=42,stratify=y)
# print(y_train.value_counts())
# print(y_test.value_counts())

'''
Đây là bộ train và bộ test , thường chia tỉ lệ 2 cái này bằng nhau để quá trình trainning phải ánh cho bộ test
chính vì thế ta với có tham số stratify áp dụng cho từng class một
career_level
senior_specialist_or_project_manager      3470
manager_team_leader                       2138
bereichsleiter                             768
director_business_unit_leader               56
specialist                                  24
managing_director_small_medium_company       3
Name: count, dtype: int64
career_level
senior_specialist_or_project_manager      868
manager_team_leader                       534
bereichsleiter                            192
director_business_unit_leader              14
specialist                                  6
managing_director_small_medium_company      1
khi có
'''
# Tại vì sao trong phần này sử dụng kĩ thuật TF , IDF bởi vì nó có dự liệu sẵn dùng nội tại còn kĩ thuật khác dữ liệu nó sẽ phần bố theo các tỉ lệ khác nhau

# print(x_train)
### Tiền xử lý dữ liệu
# Có cách nào sử dụng cột title tdf , lợi thế cho vector, word cũng thế nhé

# XỬ LÝ CỘT DESCRIPTION
vertorizer = TfidfVectorizer(lowercase=True, stop_words="english")
result = vertorizer.fit_transform(x_train["description"])
print(vertorizer.vocabulary_)
print(len(vertorizer.vocabulary_)) 
print(result.shape) # (6458, 66745) cột nhiều ta lên sử dụng


# KHI trong một cột dài ta thấy 
#unigram:(6458,66745)# có 66745 token 
#bigram:(6458, 847124)# có 847124 token

# vì sao ta thêm bigram vào nó lại nhiều như này vô lý 
# Giải thích  bỏi vì vấn đề nó lặp lại 






