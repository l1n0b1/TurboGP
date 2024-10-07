# Required Python libraries
import numpy as np
import tqdm
import time
#from six.moves import cPickle as pickle                 # To load the dataset we will be using

# TurboGP libraries
from GPIndividuals import *
from GPOperators import *
from GPflavors import *
from LowLevel import *
from GPUtils import *

from Regressor import *

# This is useful for when there is overflow, so output cells do not fill with warning messages.
# Disable to debug
import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":

    # samples
    x_training = np.linspace(-3.14, 3.14, num=100)
    x_training = np.reshape(x_training, (100,1))

    # labels
    y_training = np.sin(x_training) + np.sin(2 * x_training) + np.sin(3 * x_training) + np.sin(4 * x_training)
    y_training = y_training.flatten()

    #----------------------------------------------------#

    # We will use the same primitives Koza uses for this sample in his book
    lowlevel = ['ADD', 'SUB', 'MUL', 'DIV']
    # Now we let the Node objects know the set of input based primitives
    Node.f1_set=lowlevel
    # Set the range of constants leaf nodes can take a value from
    Node.i2_set=[-1.0,1.0]
    # Probability of choosing a primitive or ADF for a node when creating trees
    Node.prob_f1f4 = [0.8, 0.2]

    #----------------------------------------------------#

    pop_size = 4000                              # Same population as in Koza's Book
    Generations = 100
    cost_limit = 10000000                        # ten million nodes
    epochs = 1                                   # used only for minibatch training

    oper = [SimpleRegresor.mutation,              # Genetic operations to use.
            SimpleRegresor.protected_crossover]   # notice how they are defined by the type of individual we will evolve

    oper_prob = [.2, .8]                         # Since ADFs are all about crossover, we will give preference to XO over
    oper_arity = [1, 2]                          # mutation, unlike most of the rest of TurboGP examples.

    l_rate = 1.0                                 # Offspring pool size, defined as the ratio to the population size.
    minimization = True                          #
    sel_mechanism = binary_tournament            #

    #----------------------------------------------------#

    Population = []

    for _ in range(pop_size):
        Population.append(SimpleRegresor(input_vector_size=1,
                                     complexity = 9))

    #initial evaluation
    for individual in Population:
            individual.fitness(x_training, y_training)

    #----------------------------------------------------#


    diversity = []
    fitness = []
    test_fitness = []

    start = time.perf_counter()
    pbar = tqdm.tqdm(total=Generations)

    #i = 0
    for e in range(epochs):

        for j in range(Generations):

            Population, d, bf, tf = Steady_State(Population = Population,
                                                batch = x_training,
                                                labels = y_training,
                                                test_batch = None,
                                                test_labels = None,
                                                l_rate = l_rate,
                                                oper = oper,
                                                oper_prob = oper_prob,
                                                oper_arity = oper_arity,
                                                minimization = minimization,
                                                sel_mechanism = sel_mechanism,
                                                online = False)#,
                                                #pro=1)

            # count total nodels in generation
            node_count = 0
            for individual in Population:
                node_count += individual.cost()


            diversity.append(node_count)
            fitness.append(bf)
            test_fitness.append(tf)

            #i += 1
            pbar.set_postfix({'Training Fitness': bf})
            pbar.update(1)

            diversity_arr = np.asarray(diversity)
            if diversity_arr.sum() + diversity[-1] > cost_limit:
                break

    pbar.close()
    end = time.perf_counter()
    elapsed_time = int(end - start)

    #----------------------------------------------------#

    # Test it
    #test_error = gp_reg.fitness(x_test, y_test)

    # CSV format: algorithm type, train fitness, elapsed time, computational cost
    append_to_csv_row('results.csv', f"GP, {fitness[-1]}, {elapsed_time}, {diversity_arr.sum()}")
