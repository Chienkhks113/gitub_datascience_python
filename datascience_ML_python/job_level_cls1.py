import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
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
data["location"] = data["location"].apply(filter_location)
# Xử lý dữ liệu những ô bị trống
data = data.dropna()
# xử lý code dicription  
# xử kiểm tra và xử lý khi nó bị lan 
'''
data = data.dropna()
print(data.isna().sum()) 
'''
# kiểm tra xem nó có mấy sample
# print(data["career_level"].value_counts())
#Tách theo chiều dọc , tách theo chiều ngang  x sang một bên y sang một bên
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
# Tách theo chiều ngang
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2,random_state=42,stratify=y)
 # Kiểm tra xem nó có nhiều giá trị khác nhau hay không , function and  industry
print(len(x_train["industry"].unique()))

'''
Hàm print(len(x_train["industry"].unique())) 
được sử dụng để đếm và in ra số lượng các giá trị duy nhất 
trong cột "industry" của DataFrame x_train.

Cụ thể:
x_train["industry"]: Lấy cột "industry" từ DataFrame x_train.

.unique(): Phương thức này trả về một mảng chứa các giá trị duy nhất trong cột "industry".

len(...): Hàm len() đếm số lượng phần tử trong mảng được trả về bởi .unique(), tức là số lượng các giá trị duy nhất.

print(...): In ra kết quả của hàm len(...).
Ví dụ:
Giả sử bạn có DataFrame x_train với cột "industry" như sau:
import pandas as pd

data = {
    "industry": ["Tech", "Finance", "Tech", "Health", "Finance", "Health", "Education"]
}
x_train = pd.DataFrame(data)
Nếu bạn chạy:
print(len(x_train["industry"].unique()))

Kết quả sẽ là 4, bởi vì cột "industry" 
có 4 giá trị duy nhất: "Tech", "Finance", "Health", và "Education".
'''