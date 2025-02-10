import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Tạo một DataFrame mẫu
data = {'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 5, 4, 5],
        'outcome': [3, 4, 5, 6, 7]}
df = pd.DataFrame(data)

# Tính ma trận tương quan
corr_matrix = df.corr()

# # Vẽ biểu đồ nhiệt độ để trực quan hóa
# sns.heatmap(corr_matrix, annot=True)
# plt.show()
from sklearn.linear_model import LinearRegression

# Tạo mô hình
model = LinearRegression()

# Huấn luyện mô hình
model.fit(df[['feature1']], df['outcome'])

# In hệ số hồi quy
print(model.coef_)



# Dấu dương: Khi biến độc lập tăng, biến phụ thuộc cũng tăng.
# Dấu âm: Khi biến độc lập tăng, biến phụ thuộc giảm.
# Giá trị p:
# Giá trị p nhỏ hơn mức ý nghĩa (thường là 0.05): Hệ số hồi quy có ý nghĩa thống kê, tức là mối quan hệ giữa hai biến là đáng tin cậy.
# Giá trị p lớn hơn mức ý nghĩa: Hệ số hồi quy không có ý nghĩa thống kê, tức là không có bằng chứng đủ để khẳng định mối quan hệ giữa hai biến.