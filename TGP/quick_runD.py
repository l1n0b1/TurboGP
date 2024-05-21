# Quick GP example using TurboGP
# Similar example to notebook "A. Regression example",
# but in plain script file, so no jupyter needed.


import numpy as np

from genetic_program import GeneticProgramD
from Regressor import RegressorLS          # GP individual we will use
from GPUtils import tournament

# Load dataset
f = np.load("keijzer12-05pi-5000-100_online.npz", allow_pickle=True)
# Load training data
batchesX = f['batchesX']
batchesY = f['batchesY']
# Load testing data
x_testing = f['x_testing']
y_testing = f['y_testing']


# Define GP run parameters:
lowlevel = ['ADD', 'SUB', 'MUL', 'DIV', 'RELU', 'MAX', 'MEAN', 'MIN', 'X2', 'SQRT'] # Primitives
GeneticProgramD.set_primitives(lowlevel=lowlevel)

#gp_individual_class = SimpleRegresor            # Type of individual to evolve
ind_params = {'input_vector_size':2, 'complexity':12} 

oper = [RegressorLS.mutation,                 # Genetic operations to use.
        RegressorLS.protected_crossover,
        RegressorLS.mutation_i2]      # notice how they are defined by the type of individual we will evolve

oper_prob = [.4, .4, .2]                             # Probabity of each GP operation (in the same order as declared above)
oper_arity = [1, 2, 1]                              # How many parents required by each operation.


# Initialize predictor using a regressor as GP individual to evolve:
GP = GeneticProgramD(individual_class=RegressorLS , 
                    ind_params=ind_params, 
                    operations=oper, 
                    operations_prob=oper_prob, 
                    operations_arity=oper_arity, 
                    pop_size=250, 
                    epochs=2,
                    pop_dynamics="Steady_State", 
                    online=True, 
                    minimization=True,
                    sel_mechanism=tournament,
                    n_jobs=1,
                    no_populations=16, 
                    every_gen=10, 
                    top_percent=.1, 
                    topology='linear_ring')

# Train/Evolve it!

gp_reg = GP.fit(batchesX, batchesY)

# Test it on testing dataset:

print(GP.natural_score(x_testing, y_testing))