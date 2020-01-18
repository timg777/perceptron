import numpy as np
import os, sys

if __name__ == '__main__':

    try:

        # SOME TESTS

        # THERE I TAKE SAVED WEIGHTS FOR NEURON
        with open('weights.txt', 'r') as inp:
            inp = inp.readlines()
            w = np.array([float(inp[0]), float(inp[1]), float(inp[2])])

        # THERE I CREATED 2 EXAMPLES TO TEST LINEAR LOGIC NEURON
        test_1 = np.array([1, 0.2, 0.5])
        test_2 = np.array([1, 0.6, 0.8])

        # THERE I TOGGLED SUMMOTORY AND ACTIVATE FUNCTION
        print(int((test_1 @ w.T) > 0))
        print(int((test_2 @ w.T) > 0))
        print('\nsee my code to understand what is going on')

        rerun = False

    except:

        rerun = True

        # THERE I CREATED INPUT DATA TO TRAIN NEURON
        X = np.array([[1, 0.2, 0.4, 1],
                      [1, 0.6, 0.8, 0],
                      [1, 0.1, 0.3, 1],
                      [1, 0.9, 0.6, 0],
                      [1, 0.4, 0.5, 1]])

        # THERE I CREATED PRIMORDIAL WEIGHTS
        w = np.array([.0, .0, .0])

        # THERE YOU CAN SEE DUMMOTIRY FUNCTION
        def summotory(e, w):
            return e @ w.T

        # THERE YOU CAN SEE NEURON ACTIVATION
        def activation(sum_):
            return sum_ > 0

        perfect = False

        os.system('clear')

        # THERE I TRAIN NEURON
        while not perfect:
            perfect = True
            for e in X:
                # TARGET IS MY ANSWER
                target = e[-1]
                # PREDICTION IS NEURON ANSWER
                prediction = activation(summotory(e[:3], w))
                # IF MY ANSWER AND NEURON ANSWER DOESN'T SIMILAR, I SEND HIM TO RESTUDY
                if target != prediction:
                    perfect = False
                    if prediction > 0:
                        w = w - e[:3]
                    else:
                        w = w + e[:3]

                        os.system('touch weights.txt')

                        # SAVING WHEIGHT TO GET RID OF RESTUDYING NEURON
                        with open('weights.txt', 'w') as ouf:
                            ouf.write(str(w[0]))
                            ouf.write('\n')
                            ouf.write(str(w[1]))
                            ouf.write('\n')
                            ouf.write(str(w[2]))
                            ouf.write('\n')

if rerun:
    os.system('python3 perceptron.py')
else:
    sys.exit()
