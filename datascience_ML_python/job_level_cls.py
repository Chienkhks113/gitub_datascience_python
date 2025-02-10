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
data = data.dropna()
# print(data["career_level"].value_counts())
# xử lý code dicription  xử kiểm tra và xử lý khi nó bị lan 
'''
data = data.dropna()
print(data.isna().sum()) 
'''
# ho vào applice vào cái hàng đó để sử dụng cho one hot
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
chính vì thế ta với có tham số stratify áp dụng cho từng class một  , giúp để chia trong quá trình nó bằng nhau
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

# Tại vì sao trong phần này sử dụng kĩ thuật TF , IDF bởi vì nó có dữ liệu sẵn dùng nội tại còn kĩ thuật khác dữ liệu nó sẽ phần bố theo các tỉ lệ khác nhau

# print(x_train)
### Tiền xử lý dữ liệu
# Có cách nào sử dụng cột title tdf , lợi thế cho vector, word cũng thế nhé


# XỬ LÝ CỘT TITLE 
vertorizer = TfidfVectorizer(lowercase=True, stop_words="english")
result = vertorizer.fit_transform(x_train["title"])
print(vertorizer.vocabulary_)
print(len(vertorizer.vocabulary_)) #  tổng cộng 3016 token
print(result.shape)  # (6459, 3016) mảng 2 chiều , một ma trận rất là lớn

# XỬ LÝ CỘT LOCATION
encoder = OneHotEncoder()
result1 = encoder.fit_transform(x_train[["location"]]) # chú ý phần này cho mảng 2 chiều ta sử dụng riêng
print(result1.shape) # (6459, 969) ta có tổng cộng 969 hơi nhiều  sau khi ta tiền xử dữ liệu sử dụng hàm apply vào hàm nó ra ra kết quả này (6459, 94)



# XỬ LÝ CODE DESCRIPTION lại sử lý theo TF VÀ IDF tương tự như code title

'''

Cột "location" cần được đưa vào mảng 2 chiều trong dòng mã 
result1 = encoder.fit_transform(x_train[["location"]])
vì fit_transform từ nhiều encoder yêu cầu input đầu vào phải là một mảng 2 chiều.


Giải thích chi tiết:

Encoder (Bộ mã hóa): Đây có thể là các bộ mã hóa như OneHotEncoder, LabelEncoder, 

hoặc bất kỳ bộ mã hóa nào khác từ thư viện sklearn.preprocessing.

Các bộ mã hóa này thường yêu cầu dữ liệu đầu vào là một mảng 2 chiều (2D array).


Mảng 2 chiều: Mảng 2 chiều có dạng [n_samples, n_features], trong đó n_samples là số lượng mẫu (hàng)

 và n_features là số lượng đặc trưng (cột).

Trong trường hợp này, 

location là một đặc trưng và cần được đưa vào mảng 2 chiều để bộ mã hóa xử lý đúng cách.

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

# Sử dụng fit_transform với mảng 2 chiều

result1 = encoder.fit_transform(x_train[['location']])


'''






























#ONE HOT ĐƯỢC SỬ DỤNG CHO MẢNG 2 CHIỀU 
# vertorizer = TfidfVectorizer(lowercase=True , stop_words="english" , ngram_range=(1,2))
# result = vertorizer.fit_transform(x_train["title"])
# # result = vertorizer.fit_transform(x_train["description"])
# print(vertorizer.vocabulary_)
# print(len(vertorizer.vocabulary_))
# print(result.shape)
# # ách kiểm tra ma trận kết quả sau khi dùng idf , tf 
# print(result.todense())

# unigram : (6458 , 66745)
# bigram : (6458,847124)
'''
career_level
senior_specialist_or_project_manager      4338     
manager_team_leader                       2672     
bereichsleiter                             960     
director_business_unit_leader               70     
specialist                                  30     
managing_director_small_medium_company       4     
Name: count, dtype: int64
Note : Nhìn qua bộ dữ liệu ta sẽ thấy nó không cân bằng
'''
# Rèn tính kỉ luật để cho khóa sau luôn , rèn sự điềm tĩnh trong mọi lĩnh vực