# Required Python libraries
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from networkx.drawing.nx_agraph import graphviz_layout

# TurboGP libraries
from GPIndividuals import *
from GPOperators import *
from GPflavors import *
from LowLevel import *
from GPUtils import *
from Utils import *

from ADFRegressor import *          # GP individual we will use

# This is useful for when there is overflow, so output cells do not fill with warning messages.
# Disable to debug
import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":

    # Load dataset
    f = np.load("keijzer12-05pi-5000-100.npz", allow_pickle=True)
    # Load training data
    batchesX = f['batchesX']
    batchesY = f['batchesY']
    # Load testing data
    x_testing = f['x_testing']
    y_testing = f['y_testing']

    # Each of these functions must be properly defined in the corresponding python modules.
    lowlevel = ['ADD', 'SUB', 'MUL', 'DIV', 'RELU', 'MAX', 'MEAN', 'MIN', 'X2', 'SQRT']

    # Now we let the Node objects know the set of input based primitives available at each layer.
    Node.f1_set=lowlevel

    # This is the range of constants leaf nodes can take a value from (when not taking the form of a input variable)
    Node.i2_set=[-1.0,1.0]

    pop_size = 4000                                  # Population size

    oper = [Regressor1ADF.mutation,                 # Genetic operations to use.
            Regressor1ADF.protected_crossover]      # notice how they are defined by the type of individual we will evolve

    oper_prob = [.5, .5]                             # Probabity of each GP operation (in the same order as declared above)
    oper_arity = [1, 2]

    l_rate = 1.0                                     # Offspring pool size, defined as the ratio to the population size.

    minimization = True                              # if it is a minimization problem or not
    sel_mechanism = binary_tournament                # Preferred selection mechanism for parent selection

    Population = []

    for _ in range(pop_size):
        Population.append(Regressor1ADF(input_vector_size = 2,
                                        adf_arity =1,
                                        complexity = 6,
                                        adf_complexity = 6))

    #initial evaluation
    for individual in Population:
            individual.fitness(batchesX[-1], batchesY[-1])


    epochs = 2

    no_batches = len(batchesX)
    Generations = no_batches * epochs

    diversity = []
    fitness = []
    test_fitness = []

    pbar = tqdm.tqdm(total=Generations)

    for e in range(epochs):

        for j in range(no_batches):

            Population, d, bf, tf = Steady_State(Population = Population,
                                                 batch = batchesX[j],           # different minibatch each cycle
                                                 labels = batchesY[j],
                                                 test_batch = x_testing,
                                                 test_labels = y_testing,
                                                 l_rate = l_rate,
                                                 oper = oper,
                                                 oper_prob = oper_prob,
                                                 oper_arity = oper_arity,
                                                 minimization = minimization,
                                                 sel_mechanism = sel_mechanism,
                                                 online = True)

            diversity.append(d)
            fitness.append(bf)
            test_fitness.append(tf)

            pbar.update(1)

    pbar.close()

    append_to_csv_row('ADFKeijzer12.csv', test_fitness[-1])
