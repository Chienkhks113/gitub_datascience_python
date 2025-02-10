import pandas as pd
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics  import accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# Với phần này cái target là một giải liên tục

data = pd.read_csv("StudentScore.xls")
# profile = ProfileReport(data, title="Student_Score", explorative=True)
# profile.to_file("student_report.html")

# cover  thử hệ số tương quan vì thế ta lên chọn hệ số tuyến tính
print(data[["math score" , "writing score" , "reading score"]].corr())


target = "math score"
x = data.drop(target, axis = 1)
# axis loại bỏ theo cột
y=data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=5)
print(data["parental level of education"].unique()) # ordinol


 # gender : boolen , ordinal , nominal
