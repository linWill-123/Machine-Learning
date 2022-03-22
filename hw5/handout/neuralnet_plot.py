from colorsys import yiq_to_rgb
from distutils.log import error
from ftplib import all_errors
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('example',type = str)
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)


def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def one_hotY(y, n_output):
    newY = np.zeros((n_output,1))
    newY[y, 0] = 1
    return newY

def into_one_hotYs(Y, n_output):
    res = []
    for y in Y:
        newY = one_hotY(y, n_output)
        res.append(newY.ravel())
    return np.array(res)

def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))

    # Implement random initialization here
    return np.random.uniform(low = -0.1, high = 0.1, size = shape)

def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape)


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

def softmax(b):
    e = np.exp(b)
    denom = np.sum(e, axis = 0,keepdims=True)
    return e / denom

def cross_entropy(y_hat, y):
    log = np.log(y_hat)
    return -np.sum(log*y)

def mean_cross_entropy(y_hat, vectorFormY):
    log = np.log(y_hat)
    dp = np.dot(vectorFormY, log)
    diagonal = dp.diagonal()
    return -1.0/len(diagonal)*np.sum(diagonal)

def meanCrossEntropy(y_hat, vectorFormY):
    vectorFormY = vectorFormY.T
    log = np.log(y_hat)
    return -1.0/len(vectorFormY.T) * np.sum(log * vectorFormY)

# Derivative Helpers for backward:
def D_CrossEntropy(y_hat, y): 
    '''Derivative of cross entropy w.r.t b'''
    return -y + y_hat

def D_Linear(features, model, Glayer):
    '''Derivative of layer w.r.t model parameters and feature parameters'''
    Gmodel = np.matmul(Glayer, features.T)
    Gfeature = np.matmul(Glayer.T, model).T
    return Gmodel, Gfeature

def D_Sigmoid(a, z):
    return z - np.square(z)

def getS(s, Dmodel):
    return s + Dmodel*Dmodel

def update(newS, lr, epsilon, Dmodel, model):
    lrMat = lr/(np.sqrt(newS + epsilon))
    newModel = model - (lrMat * Dmodel)
    return newModel

class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size
        
        # initialize weights and biases for the models
        # alpha initialization
        shape_w1 = [hidden_size + 1, input_size]
        self.w1 = self.weight_init_fn(shape_w1)
        # beta initialization :  from hidden layer to output 
        shape_w2 = [output_size, hidden_size + 1]
        self.w2 = self.weight_init_fn(shape_w2)

        # initialize parameters for adagrad
        self.epsilon = 1e-5
        self.grad_sum_w1 = zero_init(shape_w1)
        self.grad_sum_w2 = zero_init(shape_w2)

        # feel free to add additional attributes    
        self.a = None
        self.z = None
        self.b = None 

def print_weights(nn):
    """
    An example of how to use logging to print out debugging infos.

    Note that we use the debug logging level -- if we use a higher logging
    level, we will log things with the default logging configuration,
    causing potential slowdowns.

    Note that we log NumPy matrices on separate lines -- if we do not do this,
    the arrays will be turned into strings even when our logging is set to
    ignore debug, causing potential massive slowdowns.
    :param nn: your model
    :return:
    """
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)


def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    a = np.matmul(nn.w1, X)
    z = sigmoid(a)
    z[0,:] = 1 # bias in z
    b = np.matmul(nn.w2, z)
    y_hat = softmax(b)
    # store in scope
    nn.a = a
    nn.z = z
    nn.b = b

    return y_hat


def backward(X, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
    Gb = D_CrossEntropy(y_hat, y)
    Gbeta, Gz = D_Linear(nn.z, nn.w2, Gb)
    Ga = D_Sigmoid(nn.a, nn.z)*Gz
    Galpha, Gx = D_Linear(X, nn.w1, Ga)
    return Galpha, Gbeta

def errorsAndLabels(y_hat, y):
    error = 0.0
    predictedLabels = []
    for a_hat, a_y in zip(y_hat.T, y):
        predicted_index = np.argmax(a_hat)
        label_at_index = a_y[predicted_index]
        if label_at_index != 1:
            error += 1
        # since predicted index is the predicted label, append
        predictedLabels.append(predicted_index)
    return error/len(y), predictedLabels

def test(X, Y, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    y_hat_probs = forward(X.T, nn)
    one_hotYs = into_one_hotYs(Y, nn.n_output)
    errorTest, testLabels = errorsAndLabels(y_hat_probs, one_hotYs)
    return errorTest, testLabels


def train(X_tr, y_tr, X_te, y_te, nn, all_val_mce, all_tr_mce):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    for epoch in range(nn.n_epoch):
        sampleX, sampleY = shuffle(X_tr, y_tr, epoch)
        for i in range(len(sampleX)):
            X, y = sampleX[i], sampleY[i]
            X = X[:, np.newaxis]
            y = one_hotY(y,nn.n_output)
            # forward
            y_hat = forward(X,nn)
            # baackward
            Gw1, Gw2 = backward(X,y,y_hat,nn)
            # update
            s1,s2 = nn.grad_sum_w1, nn.grad_sum_w2
            nn.grad_sum_w1 = getS(s1, Gw1)
            nn.grad_sum_w2 = getS(s2, Gw2)
            newW1 = update(nn.grad_sum_w1, nn.lr, nn.epsilon, Gw1, nn.w1)
            newW2 = update(nn.grad_sum_w2, nn.lr, nn.epsilon, Gw2, nn.w2)
            nn.w1 = newW1
            nn.w2 = newW2
        # at the end of epoch compute mean cross_entropy
        #training
        tr_preds = forward(sampleX.T,nn)
        vectorFormSampleY = into_one_hotYs(sampleY, nn.n_output)
        tr_mce = mean_cross_entropy(tr_preds, vectorFormSampleY)

        #validation
        te_preds = forward(X_te.T, nn)
        vectorFormY_te = into_one_hotYs(y_te, nn.n_output)
        te_mce = mean_cross_entropy(te_preds, vectorFormY_te)

        all_tr_mce.append(tr_mce)
        all_val_mce.append(te_mce)

    # at the end compute errors
    errorTr, TrLabels = errorsAndLabels(forward(X_tr.T, nn), into_one_hotYs(y_tr, nn.n_output))
    errorTe, TeLabels = errorsAndLabels(forward(X_te.T, nn), into_one_hotYs(y_te, nn.n_output))
    return (errorTr,TrLabels,errorTe,TeLabels)

def write_to_file(y_hat, file):
    with open(file, 'w') as f:
        np.savetxt(f, y_hat)

def write_to_metrics(trainError, testError, all_tr_mce, all_te_mce, metricsFile):
    # Write data to metrics
    with open(metricsFile, 'w') as f:
        for i in range(len(all_tr_mce)):
            f.writelines("epoch=" + str(i) + " crossentropy(train): " + str(all_tr_mce[i]) + "\n")
            f.writelines("epoch=" + str(i) + " crossentropy(validation): " + str(all_te_mce[i]) + "\n")
        f.writelines("error(train): " + str(trainError) + "\n")
        f.writelines("error(test): " + str(testError) + "\n")


def parse_metrics(file):
    data1 = []
    data2 = []
    with open(file, 'r') as f:
        i = 0
        for line in f.readlines():
            splitted = line.split(": ")
            splitted[1] = splitted[1][:len(splitted[1]) - 1]
            val = float(splitted[1])
            if i%2:
                data1.append(val)
            else:
                data2.append(val)
            i+=1
    return data1,data2

if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: You can access arguments like learning rate with args.learning_rate

    # initialize training / test data and labels
    (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics, n_epochs, n_hid, init_flag, lr) = args2data(args)
    # Build model
    if init_flag == 1:
        weight_init_fn = lambda shape: random_init(shape)
    else: 
        weight_init_fn = lambda shape: zero_init(shape)
    

    my_nn1 = NN(lr, n_epochs, weight_init_fn, len(X_tr[0]), 50, 10)
    # train model
    all_te_mce = []
    all_tr_mce = [] 
    #training result returns also predicted labels and errors
    train_res = train(X_tr, y_tr, X_te, y_te, my_nn1, all_te_mce, all_tr_mce)

    # my_nn2 = NN(lr, n_epochs, weight_init_fn, 0.01, 50, 10)
    # #training result returns also predicted labels and errors
    # avgTrainRes2, avgValRes2 = train(X_tr, y_tr, X_te, y_te, my_nn2)

    # my_nn3 = NN(lr, n_epochs, weight_init_fn, 0.001, 50, 10)
    # #training result returns also predicted labels and errors
    # avgTrainRes3, avgValRes3 = train(X_tr, y_tr, X_te, y_te, my_nn3)

    ''' FIRST
    my_nn1 = NN(lr, n_epochs, weight_init_fn, len(X_tr[0]), 5, 10)
    #training result returns also predicted labels and errors
    avgTrainRes1, avgValRes1 = train(X_tr, y_tr, X_te, y_te, my_nn1)

    my_nn2 = NN(lr, n_epochs, weight_init_fn, len(X_tr[0]), 20, 10)
    #training result returns also predicted labels and errors
    avgTrainRes2, avgValRes2 = train(X_tr, y_tr, X_te, y_te, my_nn2)

    my_nn3 = NN(lr, n_epochs, weight_init_fn, len(X_tr[0]), 50, 10)
    #training result returns also predicted labels and errors
    avgTrainRes3, avgValRes3 = train(X_tr, y_tr, X_te, y_te, my_nn3)
    my_nn4 = NN(lr, n_epochs, weight_init_fn, len(X_tr[0]), 100, 10)
    #training result returns also predicted labels and errors
    avgTrainRes4, avgValRes4 = train(X_tr, y_tr, X_te, y_te, my_nn4)
    
    my_nn5 = NN(lr, n_epochs, weight_init_fn, len(X_tr[0]), 200, 10)
    #training result returns also predicted labels and errors
    avgTrainRes5, avgValRes5 = train(X_tr, y_tr, X_te, y_te, my_nn5)
    '''
    w = 8
    h = 8
    d = 70
    
    plt.figure(figsize=(w, h), dpi=d)
    plt.title("Learning Rate = 0.1 Avg Cross Entropy Validation Graph")
    plt.xlabel('Hidden Units')
    plt.ylabel('Average Cross Entropy')
    xlabels = [5,20,50,100,200]
    
    # # avgTrainRes = [avgTrainRes1, avgTrainRes2, avgTrainRes3, avgTrainRes4, avgTrainRes5]
    # avgValRes = [avgValRes1, avgValRes2, avgValRes3, avgValRes4, avgValRes5]
    labels = [i for i in range(1,101)]
    plt.plot(labels, all_tr_mce, color = 'red', label = "Avg Training Cross Entropy")
    plt.plot(labels, all_te_mce, color = 'blue', label = "Avg Validation Cross Entropy")
    # tr, te = parse_metrics(args.example)
    # plt.plot(labels, te, color = 'green', label = 'Small Dataset Cross Entropy')
    plt.legend()
    plt.show()