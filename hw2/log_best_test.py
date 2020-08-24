import numpy as np
import sys

np.random.seed(0)
X_test_fpath = sys.argv[5]

# Parse csv files to numpy array
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)
def _preprocess(X):
    #Age
    size = X.shape[0]
    w = np.zeros((size,1))
    w2 = np.zeros((size,1))
    w3 = np.zeros((size,1))
    w4 = np.zeros((size,1))
    w5 = np.zeros((size,1))
    w6 = np.zeros((size,1))
    
    for i in range(X.shape[0]):
        if X[i][0] <= 25:
            w[i][0] = 1
        elif 25 < X[i][0] <= 35:
            w2[i][0] = 1
        elif 35 < X[i][0] <= 45:
            w3[i][0] = 1
        elif 45 < X[i][0] <= 55:
            w4[i][0] = 1
        elif 55 < X[i][0] <= 65:
            w5[i][0] = 1
        elif 65 < X[i][0]:
            w6[i][0] = 1
    X = np.append(X,w,axis = 1)
    X = np.append(X,w2,axis = 1)
    X = np.append(X,w3,axis = 1)
    X = np.append(X,w4,axis = 1)
    X = np.append(X,w5,axis = 1)
    X = np.append(X,w6,axis = 1)
    
    #Wage per hour
    w = np.zeros((size,1))
    w2 = np.zeros((size,1))
    w3 = np.zeros((size,1))
    w4 = np.zeros((size,1))
    w5 = np.zeros((size,1))
    
    for i in range(X.shape[0]):
        if X[i][126] <= 0:
            w[i][0] = 1
        elif 0 < X[i][126] <= 1200:
            w2[i][0] = 1
        elif 1200 < X[i][126] <= 1800:
            w3[i][0] = 1
        elif 1800 < X[i][126] <= 2200:
            w4[i][0] = 1
        elif 2200 < X[i][126]:
            w5[i][0] = 1
    X = np.append(X,w,axis = 1)
    X = np.append(X,w2,axis = 1)
    X = np.append(X,w3,axis = 1)
    X = np.append(X,w4,axis = 1)
    X = np.append(X,w5,axis = 1)
    
    #Capital gains
    w = np.zeros((size,1))
    w2 = np.zeros((size,1))
    w3 = np.zeros((size,1))
    w4 = np.zeros((size,1))
    
    for i in range(X.shape[0]):
        if X[i][210] <= 4600:
            w[i][0] = 1
        elif 4600 < X[i][210] <= 7600:
            w2[i][0] = 1
        elif 7600 < X[i][210] <= 15000:
            w3[i][0] = 1
        elif 15000 < X[i][210]:
            w4[i][0] = 1
    X = np.append(X,w,axis = 1)
    X = np.append(X,w2,axis = 1)
    X = np.append(X,w3,axis = 1)
    X = np.append(X,w4,axis = 1)
    
    #Capital losses
    w = np.zeros((size,1))
    w2 = np.zeros((size,1))
    w3 = np.zeros((size,1))
    w4 = np.zeros((size,1))
    w5 = np.zeros((size,1))
    
    for i in range(X.shape[0]):
        if X[i][211] <= 1400:
            w[i][0] = 1
        elif 1400 < X[i][211] <= 2000:
            w2[i][0] = 1
        elif 2000 < X[i][211] <= 2200:
            w3[i][0] = 1
        elif 2200 < X[i][211] <= 3200:
            w4[i][0] = 1
        elif 3200 < X[i][211]:
            w5[i][0] = 1
            
    X = np.append(X,w,axis = 1)
    X = np.append(X,w2,axis = 1)
    X = np.append(X,w3,axis = 1)
    X = np.append(X,w4,axis = 1)
    X = np.append(X,w5,axis = 1)
    
    #Dividends
    w = np.zeros((size,1))
    w2 = np.zeros((size,1))
    w3 = np.zeros((size,1))
    
    for i in range(X.shape[0]):
        if X[i][212] <= 0:
            w[i][0] = 1
        elif 0 < X[i][212] <= 5000:
            w2[i][0] = 1
        elif 5000 < X[i][212]:
            w3[i][0] = 1
            
    X = np.append(X,w,axis = 1)
    X = np.append(X,w2,axis = 1)
    X = np.append(X,w3,axis = 1)
    
    #Employer
    w = np.zeros((size,1))
    w2 = np.zeros((size,1))
    w3 = np.zeros((size,1))
    w4 = np.zeros((size,1))
    w5 = np.zeros((size,1))
    w6 = np.zeros((size,1))
    w7 = np.zeros((size,1))

    for i in range(X.shape[0]):
        if X[i][358] <= 0:
            w[i][0] = 1
        elif 0 < X[i][358] <= 1:
            w2[i][0] = 1
        elif 1 < X[i][358] <= 2:
            w3[i][0] = 1
        elif 2 < X[i][358] <= 3:
            w4[i][0] = 1
        elif 3 < X[i][358] <= 4:
            w5[i][0] = 1
        elif 4 < X[i][358] <= 5:
            w6[i][0] = 1
        elif 5 < X[i][358] <= 6:
            w7[i][0] = 1
    X = np.append(X,w,axis = 1)
    X = np.append(X,w2,axis = 1)
    X = np.append(X,w3,axis = 1)
    X = np.append(X,w4,axis = 1)
    X = np.append(X,w5,axis = 1)
    X = np.append(X,w6,axis = 1)
    X = np.append(X,w7,axis = 1)
    
    #Week per year
    w = np.zeros((size,1))
    w2 = np.zeros((size,1))
    w3 = np.zeros((size,1))

    for i in range(X.shape[0]):
        if X[i][507] <= 25:
            w[i][0] = 1
        elif 25 < X[i][507] <= 45:
            w2[i][0] = 1
        elif 45 < X[i][507]:
            w3[i][0] = 1
            
    X = np.append(X,w,axis = 1)
    X = np.append(X,w2,axis = 1)
    X = np.append(X,w3,axis = 1)
    return X

# Normalize training and testing data
X_test = _preprocess(X_test)

delete_list = [0, 5, 126, 127, 148, 169, 185, 186, 193, 199, 210, 211, 212, 213, 221, 255, 261, 323, 330, 331, 334, 340, 343, 350, 351, 354, 355, 358, 359, 394, 438, 481, 501, 507]
X_test = np.delete(X_test, delete_list, axis=1)
# Normalize training and testing data

best_w = np.load('best_w.npy')
best_b = np.load('best_b.npy')
# Predict testing labels
predictions = _predict(X_test, best_w, best_b)
with open(sys.argv[6].format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))
