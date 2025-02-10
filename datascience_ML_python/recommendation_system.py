import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ý tưởng nếu nó trong các bộ phim này thì ta gợi ý tương đồng như những bộ film này 


# df : datafame , num : số lượng 
def get_recommendation(user_movie, df, num):
    # lấy hết hàng cột lấy hết 
    data = df.loc[user_movie, :]
    # sắp xếp các giá trị giảm dần
    data = data.sort_values(ascending=False)[:num]
    return data

data = pd.read_csv("movie_data/movies.csv", sep="\t", encoding="latin-1", usecols=["title", "genres"])

# xử lý từng khoảng trắng của dữ liệu này cứ dấu chiều dọc ta cho nó thành khoảng trắng
data["genres"] = data["genres"].apply(lambda s: s.replace("|", " ").replace("-", ""))


# Tiền xử lý dữ liệu 
# số hóa nó biến về dạng vector

#vì sao bộ dữ liệu này ta không chia dữ liệu bộ train và bộ test
# unsi vai chia ra nó không có cái mô hình để kiểm định lên không lên chia

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["genres"])
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(tfidf_matrix.shape)





# cover nó về cái dạng bảng DataFrame nếu không cover nó sẽ không về dạng bảng nhé
# tfidf_matrix = vectorizer.fit_transform(data["genres"])
#tfidf_matrix_dense = pd.DataFrame(tfidf_matrix.todense())



# số hóa cái này giữa các bộ film
tfidf_matrix_dense = pd.DataFrame(tfidf_matrix.todense(), index=data["title"], columns=vectorizer.get_feature_names_out())

# đánh giá sự tương đồng giữa 2 bộ film


# cosine là  có u và v  cosin càng gần 1 càng tốt


# kiểm tra sự tương đồng
cosine_sim = cosine_similarity(tfidf_matrix)


# lấy tên bộ film đặt cho cả ngày và cột luôn 
cosine_sim_dense = pd.DataFrame(cosine_sim, index=data["title"], columns=data["title"])


# lấy ra top 20 hàng tương ứng , lấy ra tên hàng nào tương tự như hàng nào sử dụng trong 
top_k = 20
user_movie = "Batman Forever (1995)"
print(get_recommendation(user_movie, cosine_sim_dense, top_k))




# r2 sklearn mà âm thì nó rất là tồi nhé dự vào công thức là biết 
#  với công thức R2(y,y) nhìn chung cái phân số sẽ tử sẽ nhỏ hơn mẫu nhé 