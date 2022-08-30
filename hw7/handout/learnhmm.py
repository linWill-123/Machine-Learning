############ Welcome to HW7 ############
# TODO: Andrew-id: shaokail

'''Goal: to implement a Named Entity Recognition model which is trained with KNN 
    to correctly classify future sentences with correct tags '''

# Imports
# Don't import any other library
import argparse
import numpy as np
from utils import make_dict, parse_file
import logging

# Setting up the argument parser
# don't change anything here

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to store the hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to store the hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to store the hmm_transition.txt (B) file')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


# Hint: You might find it useful to define functions that do the following:
# 1. Calculate the init matrix
# 2. Calculate the emission matrix
# 3. Calculate the transition matrix
# 4. Normalize the matrices appropriately

def initialize_mat(word_dict, tag_dict):
    wordCount = len(word_dict)
    tagCount = len(tag_dict)
    initMat = np.ones([tagCount, 1])
    emissionMat = np.ones([tagCount, wordCount])
    transitionMat = np.ones([tagCount, tagCount])

    return initMat, emissionMat, transitionMat

def init(mat, tags, tag_dict):
    # pseudo count
    for taglist in tags:
        index = tag_dict[taglist[0]] # lookup index of corresponding tag in mat
        mat[index] += 1
    mat = mat/np.sum(mat)
    return mat

def emission(mat, sentences, tags, word_dict, tag_dict):
    # pseudo count
    for sentence,taglist in zip(sentences, tags):
        for word, tag in zip(sentence, taglist):
            row = tag_dict[tag]
            col = word_dict[word]
            mat[row,col] += 1
    return mat


def transition(mat, tags, tag_dict):
    # pseudo count
    for taglist in tags:
        for i in range(0, len(taglist) - 1):
            row = tag_dict[taglist[i]]
            col = tag_dict[taglist[i + 1]]
            mat[row,col] += 1
    return mat

def normalize(mat):
    divMat = np.sum(mat, axis = 1)[:, np.newaxis]
    mat = mat/divMat
    return mat

# TODO: Complete the main function
def main(args):

    # Get the dictionaries, looks up index to store the value in matrix
    word_dict = make_dict(args.index_to_word) # used for creating A matrix (look up of word to state probability)
    tag_dict = make_dict(args.index_to_tag) # used for creating B matrix (look up of transition probability)

    # Parse the train file
    # Suggestion: Take a minute to look at the training file,
    # it always hels to know your data :)
    sentences, tags = parse_file(args.train_input)

    logging.debug(f"Num Sentences: {len(sentences)}")
    logging.debug(f"Num Tags: {len(tags)}")
    
    
    # Train your HMM
    initMat, emissionMat, transitionMat = initialize_mat(word_dict, tag_dict) 
    # pseudocount
    initMat = init(initMat, tags, tag_dict)
    emissionMat = emission(emissionMat, sentences, tags, word_dict, tag_dict)
    transitionMat = transition(transitionMat, tags, tag_dict)   
    # normalize
    '''Note: initmat already normalizedn'''
    emissionMat = normalize(emissionMat)
    transitionMat = normalize(transitionMat)
    
    # Making sure we have the right shapes
    # logging.debug(f"init matrix shape: {init.shape}")
    # logging.debug(f"emission matrix shape: {emission.shape}")
    # logging.debug(f"transition matrix shape: {transition.shape}")


    ## Saving the files for inference
    ## We're doing this for you :)
    ## TODO: Just Uncomment the following lines when you're ready!
    
    np.savetxt(args.init, initMat)
    np.savetxt(args.emission, emissionMat)
    np.savetxt(args.transition, transitionMat)

    return 

# No need to change anything beyond this point
if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)