import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.svm import SVR as svm
from sklearn.neural_network import MLPRegressor as mlp
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LogisticRegression as logr
import seaborn as sns

# 刚才的爆红是版本函数弃用警告，不影响程序运行，可通过下面两行代码忽略
import warnings
warnings.filterwarnings('ignore')


models = ['svm', 'rf', 'mlp', 'lr', 'logr']
features = ['Temperature 1', 'Temperature 2']

df = pd.read_excel('Data.xlsx')

# cleaning
df = df.drop(df.columns[0], axis=1)

# index set
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# model choose
df = df.fillna(df.mode().iloc[0])

model = input('Choose a model: svm, rf, mlp, lr, logr: ')
feature = input('Choose a feature: Temperature 1, Temperature 2: ')
test_time = input('Please input the test time(for example 2020-01-01): ')
win = 3
feature_index = features.index(feature)

test_start = len(df[df.index < test_time])

x = []
y = []

for i in range(len(df)-win):
    x.append(df.iloc[i:i+win, feature_index].values)
    y.append(df.iloc[i+win, feature_index])
x = np.array(x)
y = np.array(y)

# training set and test set
X_train = x[:test_start]
X_test = x[test_start:]
y_train = y[:test_start]
y_test = y[test_start:]

# training
if model == 'svm':
    clf = svm()
elif model == 'rf':
    clf = rf()
elif model == 'mlp':
    clf = mlp()
elif model == 'lr':
    clf = lr(normalize=True, fit_intercept=False, n_jobs=-1)
elif model == 'logr':
    clf = logr()
# 逻辑回归就不要射那些参数了，默认的就挺好。。默认的C应该是1，这个值设置的太高可能会出现刚才的警告，其他的参数我了解也不多。

# logr要求标签y是整形，在这里做一个判断，如果model为logr，则将标签设置为整形。
if model == 'logr':
    y_train = y_train.astype('int')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# drawing
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plt.plot(y_test, label='real')
plt.plot(y_pred, label='pred')
# time set
plt.xticks(range(0, len(y_test), 50), df.index[test_start::50], rotation=45)
plt.legend()
plt.show()

pd.DataFrame(y_pred).to_csv("y_pred.csv")
pd.DataFrame(y_test).to_csv("y_test.csv")