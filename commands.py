#Recitation1.py
import numpy as np
np.random.seed(0) # setting 0 because this makes sure generate random matrix every time

f = "some link/../"
data = np.genfromtxt(f, delim="\t", dtype = None, encoding =None)
title = data[0,]
data = data[1:,]

# how to grab entire row, col in matrix?
data= np.array([[1,2,3],[4,5,6], [7,8,9]])
print("Row 0: ", data[0,:]) #row
print("Col 0: ", data[:,0])

# how to grab multiple row, col in matrix?
print("Row 0: ", data[:2,:]) #first two rows
print("Col 0: ", data[:,0])

#specifc rows:
print("Row 0: ", data[[0,2],:]) #first two rows

#to get size of array: use .shape
print(data.shape) #shape of matrix (aka 3x3 matrix) shape[0] = row, shape[1] = col
print(data.size) #number of elems (3x3 = 9)

#vectors vs arrays; arrays = axb dimension, vector = a dimension
y = [1,2,3,4]
y1 = y.reshape([1, y.shape[0]]) #1 x number of rows dimension of array
#vectors cannot be transposed; but arrays can
y1.T #transpose
np.eye(4) #identity
zeros = np.zeros([2,3])
ones = np.ones
random = np.random.random([2,3])
randomWRange = np.random.uniform(low = -0.1, high = 0,1, size =(2,3))
randomInt = np.random.randint(low = -0.1, high = 0,1, size =(2,3))

#reshape
x = axb matrices #say 4x4
x.reshape([2,8]) #can reshape into 2x8
nrows = 1
x.reshape([nrows,-1]) #the negative 1 here gives logical result of number of cols (here would be 16)

#if have two matrixes and would to combine them
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
np.hstack([x,y]) #hori
np.vstack([x,y]) #vert

#matrix add, sub, mult
x + y
x - y
4 * x
x @ y 
x * y #index by index mult

#logs and exp
np.exp(1) #e**1
np.log(1)

#max values in rows and which index found
np.max(x[0,:]) #max in 0th row
np.argmax(x[0,:]) #colindex found at 
np.min(y[:,1])
np.argmin(y[:,1])
#for two exactly sized arrays, can compute maximum/min element of each col 
np.maximum(x,y)

#sums
a.sum(axis = 0) #sum by cols
b.sum(axis = 1) #sum by rows
a.sum #entire matrix sum
a/ a.sum(axis = 0) #divide each col a by corresponding col sum 



#how to write code in a better way
#can define all the essential functions:
def read_data(input_file):
    pass

def train(x_train, y_train):
    pass

def predict(model, x_train):
    pass

def error(y_train, y_pred):
    pass
#then use separator code
if __name__ == "__main__": #this code will only run if the defined name "main" is called upon by another file 
    input_file = sys.argv[0]
    x_train, y_train = read_data(input_file)

    model = train(x_train, y_train)  #half of data sets in x, other half in y
   
    y_pred = predict(model, x_train)  #use half of data to trian???
    
    train_error = error(y_train, y_pred) #compare actual results from y_train to
'''can be useful if i wanna call the read_data function from this code but 
    don't want the bottom code (we don't have to comment out the bottom code'''

# unit test: test code right away instead of writing entire thing and test
# use pytest
# logging in python:
import logging
logging.basicConfig()
logging.debug()
logging.info()
logging.warning("hi") #print
logging.error("hi") #print`

logger = logging.setLogger()
logger.setLevel(logging.DEBUG)

#we can log output results to a file so we can save and see later