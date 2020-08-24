import sys
import pandas as pd
import numpy as np

testdata = pd.read_csv(sys.argv[1], header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    for j in range(9):
        test_x[i, j] = 0
        test_x[i, 4 * 9 + j] = 0
        test_x[i, 15 * 9 + j] = 0
        test_x[i, 16 * 9 + j] = 0
        test_x[i, 17 * 9 + j] = 0
    for j in range(1,9):
        if test_x[i, 81 + j] < 0:
            test_x[i, 81 + j] = (test_x[i, 81 + j - 1] + test_x[i, 81 + j + 1]) / 2
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)

mean_x = np.load('mean_best.npy')
std_x = np.load('std_best.npy')
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('weight_best.npy')
ans_y = np.dot(test_x, w)

import csv
with open(sys.argv[2], mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
