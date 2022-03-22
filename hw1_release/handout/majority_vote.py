# Will Lin
import sys
import numpy as np
import csv

# read the file and return as array
def read_data(in_file, linesToSkip = 1):
    filename = in_file
    data = np.genfromtxt(filename, delimiter="\t", dtype = str)
    headers = data[:linesToSkip,]
    data = data[linesToSkip:,]
    dataAsArray = np.array(data)
    return (headers,data)

# Majority Vote training: use dictionary to keep track
def train(x_train):
    countDict = dict()
    #count the majority labels in the significant column 
    for rowData in x_train:
        label = rowData[-1] #the last element of a row is the label
        if label not in countDict: #initialize label key if not already present
            countDict[label] = 1
        else:
            countDict[label] += 1
    # retrieve the keys and count from dictionary
    allLabels = list(countDict.items())
    label1, count1 = allLabels[0]
    label2, count2 = allLabels[1]

    # determine what the mojority count is
    majorityLabel = ""
    if count1 > count2: majorityLabel =  label1
    elif count1 < count2: majorityLabel = label2
    else:  #if count equal, majority label is alphabetically later label
        if label1 < label2: 
            majorityLabel =  label2 
        else: 
            majorityLabel =  label1
    #return classifier as a function
    return lambda x: majorityLabel

# predict label based on features of test data and output result to pred_outfile
def predict(model, x_train, pred_outfile):
    with open(pred_outfile, 'w') as f:
        for rowData in x_train:
            f.writelines(str(model(rowData)) + "\n")

def error(train, trainPred, test, testPred, metricsFile):
    assert(len(train) == len(trainPred))
    assert(len(test) == len(testPred))

    incorrectCountTrain = 0
    incorrectCountTest = 0
    # Compute train error
    for i in range(len(train)):
        if train[i][-1] != trainPred[i]: 
            incorrectCountTrain += 1
    trainError = float(incorrectCountTrain)/len(trainPred)

    # Compute test error
    for i in range(len(test)):
        if test[i][-1] != testPred[i]: 
            incorrectCountTest += 1
    testError = float(incorrectCountTest)/len(testPred)

    # Write data to metrics
    with open(metricsFile, 'w') as f:
        f.writelines("error(train): " + str(trainError) + "\n")
        f.writelines("error(test): " + str(testError) + "\n")

if __name__ == "__main__":
    #read command line inputs
    args = sys.argv
    trainInput = sys.argv[1]
    testInput = sys.argv[2]
    trainOutput = sys.argv[3]
    testOutput = sys.argv[4]
    metrics = sys.argv[5]

    #train model
    dataHeaders,trainDataInput = read_data(trainInput)
    model = train(trainDataInput)

    #predict train result and write predicted result to trainOutput
    predict(model, trainDataInput, trainOutput)
    _, predictedTrain = read_data(trainOutput, 0)

    #predict test result
    testHeaders, testDataInput = read_data(testInput)
    predict(model, testDataInput, testOutput)
    _, predictedTest = read_data(testOutput, 0)
    
    #metrics file compute error
    error(trainDataInput, predictedTrain, testDataInput, predictedTest, metrics)



    
    