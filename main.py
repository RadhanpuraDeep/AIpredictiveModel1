import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


heart_data = pd.read_csv('D:/heart.csv')
# print(heart_data.head())
# print(heart_data.isnull())

# print(heart_data.describe())
print(heart_data['target'].value_counts())

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
print(X)
print(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('The accuracy of training data is:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('The accuracy of testing data is:', test_data_accuracy)


input_data = (43, 0, 0, 132, 341, 1, 0, 136, 1, 3, 1, 0, 3)
input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The person does not have heart disease')
else:
    print('The person has a heart disease')

