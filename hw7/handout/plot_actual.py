'''
validation log:  -87.40364845299591
training log:  -80.15761124748151

validation log:  -80.47019720748285
training log:  -74.72168798293491

validation log:  -70.44613608272591
training log:  -67.57947125011373

validation log:  -61.0807438708335
training log:  -60.60931523798382
'''


validation = [-87.40364845299591, -80.47019720748285, -70.44613608272591, -61.0807438708335]
training = [-80.15761124748151, -74.72168798293491, -67.57947125011373, -60.60931523798382]

import matplotlib.pyplot as plt

w = 8
h = 8
d = 70

plt.figure(figsize=(w, h), dpi=d)
plt.title("Sequences length vs Training Log and Validation Log")
plt.xlabel('Sequence Number')
plt.ylabel('Log likelihood')

# # avgTrainRes = [avgTrainRes1, avgTrainRes2, avgTrainRes3, avgTrainRes4, avgTrainRes5]
# avgValRes = [avgValRes1, avgValRes2, avgValRes3, avgValRes4, avgValRes5]
labels = [10**i for i in range(1,5)]
plt.plot(labels, validation, color = 'red', label = "Avg Validation Log Likelihood")
plt.plot(labels, training, color = 'blue', label = "Avg Training Log Likelihood")

plt.legend()
plt.show()
