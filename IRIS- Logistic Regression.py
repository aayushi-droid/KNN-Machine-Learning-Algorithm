import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load The Data Into DataFrame with Pandas 
iris = load_iris()
X = pd.DataFrame(iris.data)    # Independent Variable
y = pd.DataFrame(iris.target)  # Dependent Variable

#print(X.head())      # print 5 rows of independent variable
#Label Encoder
encode = LabelEncoder()
y = encode.fit_transform(y)

# convert into train and test data
trainX,testX,trainy, testy = train_test_split(X, y, test_size= 0.2)

# fit and predict model
model = LogisticRegression().fit(trainX,trainy)
predy = model.predict(testX)

# check accuracy score
score = accuracy_score(testy, predy)
print(f'Accuracy Score : {score}')


