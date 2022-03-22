import numpy as np
import sys

def load_formatted_data(infile):
    dataset = np.loadtxt(infile, comments=None, encoding='utf-8',
                         dtype= 'float')
    return dataset

def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def h(theta, X, i):
    return sigmoid(np.dot(theta, X[i]))

def dJ(theta, X, y, i):
    gradient = (y[i] - h(theta, X, i))*X[i]
    return gradient

def train(theta, X, y, num_epoch, learning_rate):
    X[:, 0] = 1 #initialize intercept col with 1's; this replaced where labels were
    for epoch in range(num_epoch):
        for i in range(len(X)):
            theta += learning_rate * dJ(theta, X, y, i)
    return theta
            
def predict(theta, X):
    dot_weights = np.matmul(X, theta) #compute dp of each X(i) w.r.t theta
    y = sigmoid(dot_weights) #compute prob of each X(i) w.r.t theta
    y = np.where(y >= 0.5, 1, 0) #replace where prob > 0.5
    return y

def compute_error(y_pred, y):
    return np.sum(y != y_pred)/len(y_pred)

def write_to_file(preds, file):
    with open(file, 'w') as f:
        for pred in preds:
            f.writelines(str(int(pred)) + "\n")

def write_to_metrics(trainError, testError, metricsFile):
    # Write data to metrics
    with open(metricsFile, 'w') as f:
        f.writelines("error(train): " + str(trainError) + "\n")
        f.writelines("error(test): " + str(testError) + "\n")

if __name__ == '__main__':
    formatted_train_in = sys.argv[1] 
    formatted_validation_in = sys.argv[2] 
    formatted_test_in = sys.argv[3] 
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    num_epoch = int(sys.argv[7])
    learning_rate = float(sys.argv[8])

    #init model params
    X = load_formatted_data(formatted_train_in)
    y = np.copy(X[:,0]) #labels col
    theta = np.zeros(len(X[0]))#size of theta = number of features + intercept
    model = train(theta, X, y, num_epoch, learning_rate)
    print(model)
    # predictions with training
    y_pred = predict(model, X)
    train_error = compute_error(y_pred, y)
    write_to_file(y_pred, train_out)
    # predictions with test
    testX = load_formatted_data(formatted_test_in)
    y = np.copy(testX[:,0])
    y_pred = predict(theta, testX)
    test_error = compute_error(y_pred, y)
    write_to_file(y_pred, test_out)

    write_to_metrics(train_error, test_error, metrics_out)