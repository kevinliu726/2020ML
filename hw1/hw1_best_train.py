import sys
import pandas as pd
import numpy as np

data = pd.read_csv(sys.argv[1], encoding = 'big5')
data = data.iloc[:,3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    sample[0,:] = 0
    sample[4,:] = 0
    sample[15,:] = 0
    sample[16,:] = 0
    sample[17,:] = 0
    for i in range(1,479):
        if sample[9,i] < 0:
            sample[9,i] = (sample[9,i - 1] + sample[9, i + 1]) / 2
    month_data[month] = sample

x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]

mean_x = np.mean(x, axis = 0) #18 * 9
std_x = np.std(x, axis = 0) #18 * 9
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
np.save('std_best.npy', std_x)
np.save('mean_best.npy', mean_x)
import math
x_train_set = x[: math.floor(len(x) * 1), :]
y_train_set = y[: math.floor(len(y) * 1), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]

dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
learning_rate = 7
iter_time = 50000
adagrad = np.zeros([dim, 1])
eps = 1e-7
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight_best.npy', w)
