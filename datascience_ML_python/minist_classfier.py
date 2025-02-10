import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

with np.load("mnist.npz" , "r") as f:
    # print(f["x_train"].shape)
    # print(f["x_test"].shape)
    # print(f["y_train"].shape)
    # print(f["y_test"].shape)
    x_train , x_test = f["x_train"] , f["x_test"]
    y_train , y_test = f["y_train"] , f["y_test"]


# print(x_train.shape , y_train.shape)
# print(x_test.shape , y_test.shape)


# phải có bước này thì nó mới chạy được nhé
x_train = np.reshape(x_train , (x_train.shape[0] , -1))
x_test = np.reshape(x_test , (x_train.shape[0] , -1))
# print(x_train.shape)
# print(x_test.shape)
# với train (6000 , 784 ) 60000 sample và mỗi sample nó có 784

# càng gần 255 là màu trắng


model = Pipeline( steps = [
    ("scaler" , StandardScaler()) , 
    ("model" , DecisionTreeClassifier())
]

)
model.fit(x_train , y_train)
y_predict = model.predict(x_test)
print(classification_report(y_test , y_predict))

# xem va fix lai