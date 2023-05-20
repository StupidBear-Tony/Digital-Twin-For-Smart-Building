import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import SVC as svm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.linear_model import LogisticRegression as logr
from sklearn.linear_model import LinearRegression as lr

models = ['svm', 'rf', 'mlp', 'lr', 'logr']

features = ['date', 'mels_S','lig_S','mels_N','hvac_N','hvac_S']

model = input('Choose a model: svm, rf, mlp, lr, logr: ')
feature = input('Choose a feature: date, mels_S, lig_S, mels_N, hvac_N, hvac_S: ')
test_time = input('Please input the test time(for example 2020-01-01): ')
win = 3
feature_index = features.index(feature)

# data loading
df = pd.read_csv('data/Bldg59_clean data/ele.csv')
df = df.iloc[:, :-1]
df['date'] = pd.to_datetime(df['date'])

# preliminary data processing
df = df.fillna(df.mode().iloc[0])
for date in features[1:]:
    df[date] = df[date].astype('int')#



test_start = len(df[df['date'] < test_time])
test_end = len(df[df['date'] < test_time[:-1] + str(int(test_time[-1])+1)])
print(test_time, test_time[:-1] + str(int(test_time[-1])+1))
x = []
y = []


for i in range(len(df)-win):
    x.append(df.iloc[i:i+win, feature_index].values)
    y.append(df.iloc[i+win, feature_index])
x = np.array(x)
y = np.array(y)


# classification of training dataset and testing dataset
X_train = x[:test_start]
X_test = x[test_start:test_end]
y_train = y[:test_start]
y_test = y[test_start:test_end]


# training model
clf = eval(model+'()')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

xticks = df['date'][test_start:test_end]
pd.DataFrame(y_test).to_csv("y_test.csv")
pd.DataFrame(y_pred).to_csv("y_pred.csv")
pd.DataFrame(y_pred).to_csv("y_train.csv")





