import numpy as np
import sys

np.random.seed(0)
X_train_fpath = sys.argv[3]
Y_train_fpath = sys.argv[4]

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    #if specified_column == None:
    #    specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

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
X_train = _preprocess(X_train)
X_test = _preprocess(X_test)

delete_list = [0, 5, 126, 127, 148, 169, 185, 186, 193, 199, 210, 211, 212, 213, 221, 255, 261, 323, 330, 331, 334, 340, 343, 350, 351, 354, 355, 358, 359, 394, 438, 481, 501, 507]
X_train = np.delete(X_train, delete_list, axis=1)
X_test = np.delete(X_test, delete_list, axis=1)
# Normalize training and testing data

dev_ratio = 0.2
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

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
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

# Zero initialization for weights ans bias
w = np.zeros((data_dim,))
b = np.zeros((1,))
best_w = (w,b,-1)
ada_w = np.zeros((data_dim,))
ada_b = np.zeros((1,))

# Some parameters for training
max_iter = 500
batch_size = 15
learning_rate = 2e-2

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# Iterative training
for epoch in range(max_iter):
    # Random shuffle at the begging of each epoch
    X_train, Y_train = _shuffle(X_train, Y_train)
        
    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[idx*batch_size:(idx+1)*batch_size]

        # Compute the gradient
        w_grad, b_grad = _gradient(X, Y, w, b)

        ada_w += w_grad ** 2
        ada_b += b_grad ** 2
        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate/np.sqrt(ada_w + 1e-7) * w_grad
        b = b - learning_rate/np.sqrt(ada_b + 1e-7) * b_grad
            
    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    acc_dev = _accuracy(Y_dev_pred, Y_dev)
    acc_loss = _cross_entropy_loss(y_dev_pred, Y_dev) / dev_size
    if acc_dev > best_w[-1]:
        best_w = (w,b,acc_dev)
    dev_acc.append(acc_dev)
    dev_loss.append(acc_loss)

print('Development loss: {}'.format(dev_loss[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

# Predict testing labels
np.save("best_w", best_w[0])
np.save("best_b", best_w[1])
