import csv
import sys
import numpy as np

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################



def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An N x 2 np.ndarray. N is the number of data points in the tsv file. The
        first column contains the label integer (0 or 1), and the second column
        contains the movie review string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

def bag_of_word(dataset, d, outfile):
    # we know d elems are {word, index}
    res = []
    vectorsize = len(d.keys())
    with open(outfile, 'w') as f:
        for i in range(len(dataset)):
            label, review = dataset[i][0], dataset[i][1]
            vectorizedString = np.zeros(vectorsize)
            for word in review.split(" "):
                if word in d:
                    vectorizedString[d[word]] = 1
            f.write(str(int(label)) + "\t")
            for val in vectorizedString:
                f.write(str(int(val)) + "\t")
            f.write("\n")
def word_embedding(dataset, d, featuresize, outfile):
    '''idea: to give each review a rating by sum of positive and negative words: if review has more positive value perhaps
        we view it as positive'''
    # res = []
    with open(outfile, 'w') as f:
        for i in range(len(dataset)):
            label, review = dataset[i][0], dataset[i][1]
            # keep track of number of words that are in dictionary so we can mult by avg value
            count = 0 
            # each vector consists of len = number of features
            vectorArray = np.zeros(featuresize)
            for word in review.split(" "):
                if word in d:
                    vectorArray += d[word]
                    count += 1
            if count: #if we have words in dict then divide by count
                vectorArray = vectorArray*(1/count)
            strArray = np.array2string(vectorArray, precision = 6, separator = "\t")
            f.writelines(str("{:.6f}".format(label)) + "\t" + strArray[1:len(strArray) - 1] + "\n")

def are_rows_same_size(file):
    dataset = np.loadtxt(file, encoding='utf-8',
                         dtype= None)
    length = len(dataset[0])
    for data in dataset[1:]:
        assert(length == len(data))

if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3] 
    dict_input = sys.argv[4] 
    feature_dictionary_input = sys.argv[5] 
    formatted_train_out = sys.argv[6] 
    formatted_validation_out = sys.argv[7] 
    formatted_test_out = sys.argv[8] 
    feature_flag = int(sys.argv[9])

    trainData = load_tsv_dataset(train_input)
    valData = load_tsv_dataset(validation_input)
    testData = load_tsv_dataset(test_input)

    if feature_flag == 1:
        #load dictionary 
        d = load_dictionary(dict_input)
        #train
        bag_of_word(trainData, d, formatted_train_out)
        bag_of_word(valData, d, formatted_validation_out)
        bag_of_word(testData, d, formatted_test_out)
    else: #feature_flag == 2
        d = load_feature_dictionary(feature_dictionary_input)
        numFeatures = len(list(d.values())[0])
        #train
        word_embedding(trainData, d, numFeatures, formatted_train_out)
        word_embedding(valData, d, numFeatures, formatted_validation_out)
        word_embedding(testData, d, numFeatures, formatted_test_out)

    # are_rows_same_size(formatted_train_out)