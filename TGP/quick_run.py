# Quick GP example using TurboGP
# Similar example to notebook "A. Regression example",
# but in plain script file, so no jupyter needed.


import numpy as np

from genetic_program import GeneticProgram
from Regressor import SimpleRegresor          # GP individual we will use
from GPUtils import binary_tournament

# Load dataset
f = np.load("linobi0-pi-1000-100.npz", allow_pickle=True)
# Load training data
batchesX = f['batchesX']
batchesY = f['batchesY']
# Load testing data
x_testing = f['x_testing']
y_testing = f['y_testing']


# Define GP run parameters:
lowlevel = ['ADD', 'SUB', 'MUL', 'DIV', 'X2', 'MAX', 'MEAN', 'MIN', 'RELU'] # Primitives
GeneticProgram.set_primitives(lowlevel=lowlevel)

#gp_individual_class = SimpleRegresor            # Type of individual to evolve
ind_params = {'input_vector_size':1, 'complexity':7} 

oper = [SimpleRegresor.mutation,                 # Genetic operations to use.
        SimpleRegresor.protected_crossover]      # notice how they are defined by the type of individual we will evolve

oper_prob = [.5, .5]                             # Probabity of each GP operation (in the same order as declared above)
oper_arity = [1, 2]                              # How many parents required by each operation.


# Initialize predictor using a regressor as GP individual to evolve:
GP = GeneticProgram(individual_class=SimpleRegresor , 
                    ind_params=ind_params, 
                    operations=oper, 
                    operations_prob=oper_prob, 
                    operations_arity=oper_arity, 
                    pop_size=1000, 
                    epochs=10,
                    pop_dynamics="Steady_State", 
                    online=True, 
                    minimization=True,
                    n_jobs=4)

# Train/Evolve it!

gp_reg = GP.fit(batchesX, batchesY)

# Test it on testing dataset:

print(GP.natural_score(x_testing, y_testing))
