############ Welcome to HW7 ############
# TODO: Andrew-id: 


# Imports
# Don't import any other library
from audioop import avg
import numpy as np
from utils import make_dict, parse_file, get_matrices, write_predictions, write_metrics
import argparse
import logging

# Setting up the argument parser
# don't change anything here
parser = argparse.ArgumentParser()
parser.add_argument('validation_input', type=str,
                    help='path to validation input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to the learned hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to the learned hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to the learned hmm_transition.txt (B) file')
parser.add_argument('prediction_file', type=str,
                    help='path to store predictions')
parser.add_argument('metric_file', type=str,
                    help='path to store metrics')                    
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


# Hint: You might find it helpful to define functions 
# that do the following:
# 1. Calculate Alphas
# 2. Calculate Betas
# 3. Implement the LogSumExpTrick
# 4. Calculate probabilities and predictions
def logSumExpTrick(v):
    m = max(v)
    return m + np.log(np.sum(np.exp(v-m)))

def minBayes(t, alpha, beta, tag_dict):
    '''returns minimum bayes prediction for t-th word'''
    key_list = list(tag_dict.keys())
    val_list = list(tag_dict.values())

    index = np.argmax(alpha[t]+beta[t])
    
    return key_list[val_list.index(index)]

# TODO: Complete the main function
def main(args):
    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)

    # Parse the validation file
    sentences, tags = parse_file(args.validation_input)

    ## Load your learned matrices
    ## Make sure you have them in the right orientation
    ## TODO:  Uncomment the following line when you're ready!
    
    init, emission, transition = get_matrices(args)

    # TODO: Conduct your inferences

    # TODO: Generate your probabilities and predictions

    # TODO: store your predicted tags here (in the right order)
    predicted_tags = []
    # TODO: store your calculated average log-likelihood here
    avg_log_likelihood = []

    for sequence in sentences:
        # each sequence needs to run forward backward
        dpAlpha = [None for _ in range(len(sequence))]
        dpBeta = [None for _ in range(len(sequence))]
        '''Foreward'''
        for t in range(len(sequence)): #for t in T
            if t != 0:
                emmisionIndex = word_dict[sequence[t]]
                A = np.log(emission[:,emmisionIndex])
                B = np.log(transition)
                logged_alpha = dpAlpha[t-1]
                W = B.T + logged_alpha
                newW = []
                for v in W:
                    newW.append(logSumExpTrick(v))
                newW = np.array(newW)
                newW += A
                alpha_t = newW

            else: # initialization
                emmisionIndex = word_dict[sequence[t]]
                A = np.log(emission[:,emmisionIndex])
                pi = np.log(init)
                W = A + pi
                alpha_t = W
            dpAlpha[t] = alpha_t
        '''Backward'''
        for t in range(len(sequence) - 1, -1, -1):
            if t != len(sequence) - 1:
                emmisionIndex = word_dict[sequence[t+1]]
                A = np.log(emission[:,emmisionIndex])
                B = np.log(transition)
                W = A + dpBeta[t + 1] + B
                newW = []
                for v in W:
                    newW.append(logSumExpTrick(v))
                beta_t = np.array(newW)
            else:
                W = np.zeros(len(tag_dict))
                beta_t = W
            dpBeta[t] = beta_t
        # compute predicted tags and log likelihoods
        log_prob = logSumExpTrick(dpAlpha[len(dpAlpha) - 1])
        avg_log_likelihood.append(log_prob)
        # prediction
        predicted_states = []
        for t in range(len(sequence)):
            
            predicted_tag = minBayes(t, dpAlpha, dpBeta, tag_dict)
            predicted_states.append(predicted_tag)
        predicted_tags.append(predicted_states)

    
    accuracy = 0 # We'll calculate this for you
    avg_log_likelihood = sum(avg_log_likelihood)/len(avg_log_likelihood)
    ## Writing results to the corresponding files.  
    ## We're doing this for you :)
    ## TODO: Just Uncomment the following lines when you're ready!

    accuracy = write_predictions(args.prediction_file, sentences, predicted_tags, tags)
    write_metrics(args.metric_file, avg_log_likelihood, accuracy)

    return

if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)
