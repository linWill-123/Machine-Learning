import argparse
import numpy as np
from environment import MountainCar, GridWorld
import sys
import random

# NOTE: We highly recommend you to write functions for...
# - converting the state to a numpy array given a sparse representation
# - determining which state to visit next


# retrieve the dictionary telling where the state is occupied (which pos used)
def stateToNp(d,sizeS):
    vector = np.zeros(sizeS)
    for key,val in d.items():
        vector[key] = val
    return vector


def main(args):
    # Command line inputs
    mode = args.mode
    weight_out = args.weight_out
    returns_out = args.returns_out
    episodes = args.episodes
    max_iterations = args.max_iterations
    epsilon = args.epsilon
    gamma = args.gamma
    learning_rate = args.learning_rate
    debug = args.debug

    # We will initialize the environment for you:
    if args.environment == 'mc':
        env = MountainCar(mode=mode, debug=debug)
    else:
        env = GridWorld(mode=mode, debug=debug)

    # TODO: Initialize your weights/bias here
    weights =  [[0 for _ in range(env.state_space)] for _ in range(env.action_space)]# Our shape is |A| x |S|, if this helps.
    bias = 0
    # If you decide to fold in the bias (hint: don't), recall how the bias is
    # defined!

    returns = []  # This is where you will save the return after each episode
    t = 0
    for episode in range(episodes):
        # Reset the environment at the start of each episode
        state = env.reset()  # `state` now is the initial state
        state = stateToNp(state, env.state_space)
        total_return = 0
        for it in range(max_iterations):
            # TODO: Fill in what we have to do every iteration
            # Hint 1: `env.step(ACTION)` makes the agent take an action
            #         corresponding to `ACTION` (MUST be an INTEGER)
            # Hint 2: The size of the action space is `env.action_space`, and
            #         the size of the state space is `env.state_space`
            # Hint 3: `ACTION` should be one of 0, 1, ..., env.action_space - 1
            # Hint 4: For Grid World, the action mapping is
            #         {"up": 0, "down": 1, "left": 2, "right": 3}
            #         Remember when you call `env.step()` you have to pass
            #         the INTEGER representing each action!
            

            # get q_values for all possible actions 
            q_values = np.dot(weights,state) + bias

            # Decide to exploit or explore
            do_greedy = random.random() < (1 - epsilon)

            if do_greedy: # exploit
                # select action with highest q-val
                action = np.argmax(q_values)
            else: # explore
                action = random.randrange(0,env.action_space)
                
            # step
            next_state, r, done = env.step(action)
            next_state = stateToNp(next_state, env.state_space)
            
            # update theta matrix
            Q_next_state = np.dot(weights,next_state) + bias
            TD_target = r + gamma*np.max(Q_next_state)
            TD_error = q_values[action] - TD_target # original q - new q
            
            weights[action] = weights[action] - learning_rate*TD_error*state

            # update bias
            DJDb = TD_error
            bias -= learning_rate*DJDb
            
            # transition state
            state = next_state
            total_return += r # immediate reward
            if done:
                break
        
        returns.append(total_return)
    
    rolling_mean = []
    for i in range(0, episodes - 25 + 1, 25):
        rolling_mean.append(sum(returns[i : i + 25])/25)   

    import matplotlib.pyplot as plt

    w = 8
    h = 8
    d = 70

    plt.figure(figsize=(w, h), dpi=d)
    plt.title("Return v.s. Episodes")
    plt.xlabel('Episodes')
    plt.ylabel('Return')

    labels = [i for i in range(1, episodes + 1)]
    plt.plot(labels, returns, color = 'red', label = "Return at Episode")
    rollingMeansLabel = [i for i in range(25, episodes + 1, 25)]
    plt.plot(rollingMeansLabel, rolling_mean, color = 'blue', label = "Rolling Means over previous 25 episodes")

    plt.legend()
    plt.show()
        
    

if __name__ == "__main__":
    # No need to change anything here
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', type=str, choices=['mc', 'gw'],
                        help='the environment to use')
    parser.add_argument('mode', type=str, choices=['raw', 'tile'],
                        help='mode to run the environment in')
    parser.add_argument('weight_out', type=str,
                        help='path to output the weights of the linear model')
    parser.add_argument('returns_out', type=str,
                        help='path to output the returns of the agent')
    parser.add_argument('episodes', type=int,
                        help='the number of episodes to train the agent for')
    parser.add_argument('max_iterations', type=int,
                        help='the maximum of the length of an episode')
    parser.add_argument('epsilon', type=float,
                        help='the value of epsilon for epsilon-greedy')
    parser.add_argument('gamma', type=float,
                        help='the discount factor gamma')
    parser.add_argument('learning_rate', type=float,
                        help='the learning rate alpha')
    parser.add_argument('--debug', type=bool, default=False,
                        help='set to True to show logging')
    main(parser.parse_args())
