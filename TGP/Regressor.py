#-----------------------------------------------------------------------------------#
# This file implements a GP-individual based on a single tree and designed to carry #
# scalar regression. It can be considered the most basic example of GP-individuals  #
# in the TurboGP library. It is pretty straighforward in its implementation: a tree #
# that receives as input a n-size vector and outputs a single real number, that is, #
# the prediction.                                                                   #
#
# This python class exemplifies how individuals must incorporate their own genetic  #
# operations (crossover, mutation, etc.) methods that encapsulate the methods found #
# in the GPOperators.py file, as well as provide protections in the case of cross-  #
# over (such as grammatical validity ensurance, and max tree depth protection).     #
#
# The basic steps that a script must perform in order to evolve a regressor with    #
# TurboGP library, is to create a bunch of individuals of the class herein defined  #
# (the population) and pass them through one of the population dynamics implemented #
# in GPflavors.py file, for the desired number of generations or epochs (as well as #
# providing a dataset, of course).                                                  #
#
# For a complete guide on the usage of this GPIndividual class, and the TurboGP     #
# library as a whole, refer to the jupyter notebooks provided as examples, as well  #
# as to the GPSimpleReg.py script, also shipped along the rest of the files of the  #
# library.                                                                          #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, Morales-Reyes, HJ Escalante. 2020.                       #
# Instituto Nacional de Astrofisica, Optica y Electronica (INAOE). Puebla, Mexico   #
# Development funded by CONACyT grant No. 436184, CONCyTEP grant 2019-52D, and INAOE#
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#


from GPIndividuals import *
from GPOperators import *

import numpy as np
import random


class SimpleRegresor:
    ''' This class implements a GPtree based regressor. In consists of a tree object (as defined in GPIndividual.py),
    a fitness value, functions that allow to perform a prediction (predict), evaluate the performance of the individual
    against a single sample and its label (case_fitness), evaluate the performance of the individual against a bunch
    of labeled samples (fitness), and methods that implement genetic operations applicable to this type of individual.'''

    def __init__(self, input_vector_size, complexity, grow_method='variable'):
        '''This is the constructor method for the SimpleRegresor class. In order to create and initialize an individual
        of this class, you have to define the size of the input vectors (input_vector_size), the max tree depth allowed
        (complexity), and the grow method.'''

        # the feature variables input set (i1) consist of a collection of integers.
        input_set = np.arange(input_vector_size)

        # Initialize tree
        self.tree = Tree(max_tree_depth = complexity,
                         i1_set = input_set,
                         grow_method = grow_method)

        # grow it
        self.tree.grow_random_tree()
        
        while self.tree.nodes[0].subtree_depth < 2:
            # Initialize tree
            self.tree = Tree(max_tree_depth = complexity,
                             i1_set = input_set,
                             grow_method = grow_method)

            # grow it
            self.tree.grow_random_tree()

        # SimpleRegresor uses MSE as error measure. Finding an acceptable or optimal SimpleRegresor is a minimization
        # problem; thefore, fitness value is initialized to inf.
        self.fitness_value = float('Inf')

    def predict(self, instance):
        '''This function applies the individual to a sample (instance) in order to perform a prediction. '''

        result = self.tree.evaluate(instance)

        return result

    def case_fitness(self, instance, label):
        ''' Test performance of the individual against a single instance.'''

        output = self.predict(instance)

        # Return quadratic error.
        result = ((label - output) ** 2)

        return result

    def fitness(self, samples_matrix, labels_matrix):
        ''' This function calculates the actual fitness of the individual, for a given dataset. SimpleRegresor uses the
        MSE as error measure, so the fitness value is calculated by averaging the case fitnesses obtained across all
        samples of the provided dataset.'''

        batch_size = len(samples_matrix)

        case_fitnesses = []

        for i in range(batch_size):
            case_fitnesses.append(self.case_fitness(samples_matrix[i], labels_matrix[i]))

        self.fitness_value = np.asarray(case_fitnesses).mean()

        return self.fitness_value

    @staticmethod
    def crossover(filter1, filter2):
        ''' This is a static method of the SimpleRegressor class that defines standard subtree crossover operation for
        this type of GP individuals. It is basically a wrapper for subtree_crossover method found in GPOperators.py.
        It receives as input two SimpleRegresor individuals and returns as output two new individuals. This version of
        crossover is not protected in any way, neither gramatically nor in max tree depth. It does nothing more than
        choosing random nodes in both trees as exchange points, and call subtree_crossover method on them. Therefore it
        should be used only when just low-level primitves are enabled (i.e. no mezzanine functions), and it may induce
        bloating in indivduals as generations elapse, making difficult to predict when a GP run will finish (runs may
        extend in time indefinitely).'''

        upto1 = len(filter1.tree.nodes)
        upto2 = len(filter2.tree.nodes)
        
        probs1 = []
        for i in range(len(filter1.tree.nodes)):
            if i == 0:
                probs1.append(.0)
            else:
                if filter1.tree.nodes[i].node_type in ['i1', 'i2', 'i3', 'i5']:
                    probs1.append(.1)
                else:
                    probs1.append(.9)
        
        probs1 = np.asarray(probs1)
        probs1 = probs1/probs1.sum()
        
        probs2 = []
        for i in range(len(filter2.tree.nodes)):
            if i == 0:
                probs2.append(.0)
            else:
                if filter2.tree.nodes[i].node_type in ['i1', 'i2', 'i3', 'i5']:
                    probs2.append(.1)
                else:
                    probs2.append(.9)
        
        probs2 = np.asarray(probs2)
        probs2 = probs2/probs2.sum()
        
                    

        # Notice Crossover protection from picking root nodes
        node1 = np.random.choice(a=np.arange(upto1), p=probs1)      
        node2 = np.random.choice(a=np.arange(upto2), p=probs2)      

                

        offspring1 = deepcopy(filter1)
        offspring2 = deepcopy(filter2)

        offspring1.tree, offspring2.tree = subtree_crossover(filter1.tree, node1, filter2.tree, node2)

        return offspring1, offspring2

    @staticmethod
    def mutation(filter1):
        ''' This is a static method of the SimpleRegressor class that defines subtree mutation operation for such kind
        of individuals. It is basically a wrapper for subtree_mutation method defined in GPOperators.py. It receives as
        input one individual and return as output one new individual where one randomly picked subtree has been altered.
        Since subtree_mutation is protected in every sense (both max depth, and gramatically), this function can be used
        whatever types of primitives are enables, and will always yield trees that respect the max tree depth parameter,
        resulting in GP runs with a predicatble time-cost behavior.'''

        upto = len(filter1.tree.nodes)

        node = np.random.randint(upto)

        offspring = deepcopy(filter1)

        offspring.tree = subtree_mutation(filter1.tree, node)

        return offspring

    @staticmethod
    def protected_crossover(filter1, filter2):
        ''' This is a static method of the Simple Regressor class of individuals that defines a protected crossover type
        of operation. This type of crossover ensures that generated offspring trees do not exced the max allowed depth.
        To be able to perform such operation, it is required to perform a pseudo crossover actually: the genetic material
        (subtrees) are not actually swapped between trees, instead, the first child is generated by randomly picking a
        node the first parent and then search for a subtree of a valid depth on the second parent. The process is then
        repeated switching parents role, in order to generate the second offspring. Notice how while this operation
        protect individuals from exceeding the max allowed tree, it does not verify that gramatically valid individuals
        are generated, therefore, just as crossover method defined above, it should be used with only low-level primiti-
        ves enabled.

        This is the recommended crossover operation to use (rather than plain crossover), since guarantees
        that individuals generated will be within a depth boundary, resulting in GP runs that complete in predictable
        time spans, for any number of evolutionary cycles.

        For a variant of this operation that also guarantees grammatical validation, see NonConvFilter class of indivi-
        duals, defined in files NonConvolutionalMezzanineFilter.py and NonConvolutionalHighFilter.py'''

        # First offspring

        # pick node in first parent completely randomly
        node1 = np.random.randint(1, len(filter1.tree.nodes))

        # Get destiny node depth:
        origin_depth = filter1.tree.nodes[node1].current_tree_depth

        # Search for a valid subtree in second parent
        node2 = np.random.randint(1, len(filter2.tree.nodes))

        # Get subtree depth
        source_subtree_depth = filter2.tree.nodes[node2].subtree_depth

        # Verify if is a valid subtree to import
        while origin_depth + source_subtree_depth > filter1.tree.max_tree_depth:

            # Search for a valid subtree in second offspring
            node2 = np.random.randint(1, len(filter2.tree.nodes))

            # Get subtree depth
            source_subtree_depth = filter2.tree.nodes[node2].subtree_depth

        # Now perform first crossover

        offspring1 = deepcopy(filter1)

        offspring1.tree, _ = subtree_crossover(filter1.tree, node1, filter2.tree, node2)


        # Second Offspring

        # pick node in second parent completely randomly
        node2 = np.random.randint(1, len(filter2.tree.nodes))

        # Get destiny node depth:
        origin_depth = filter2.tree.nodes[node2].current_tree_depth

        # Search for a valid subtree in first parent
        node1 = np.random.randint(1, len(filter1.tree.nodes))

        # Get subtree depth
        source_subtree_depth = filter1.tree.nodes[node1].subtree_depth

        # Verify if is a valid subtree to import
        while origin_depth + source_subtree_depth > filter2.tree.max_tree_depth:

            # Search for a valid subtree in first parent
            node1 = np.random.randint(1, len(filter1.tree.nodes))

            # Get subtree depth
            source_subtree_depth = filter1.tree.nodes[node1].subtree_depth

        # Now perform first crossover

        offspring2 = deepcopy(filter2)

        _, offspring2.tree = subtree_crossover(filter1.tree, node1, filter2.tree, node2)

        return offspring1, offspring2

    @staticmethod
    def samedepths_crossover(filter1, filter2):
        ''' This is a static method of the SimpleRegressor class of individuals that defines the SameDepths subtree
        crossover, promoted by Kim Harries & Peter Smith (1997). SameDepths crossover is a type of protected crossover.
        It works first by determining which of both parents is shallower and uses its depth as a range to pick at
        random a depth, then searches for nodes within such given depth and performs the subtree crossover.

        Protected crossover and SameDepths crossover can produce better results than one another, depending on the
        particular application, but in general both yield good results. They may as well be used in tandem to sometimes
        produce even better results.

        Like protected_crossover, the method here implemented does not perform grammatical validations, therefore should
        be used with low-level only primitives.'''



        upto1 = filter1.tree.nodes[0].subtree_depth
        upto2 = filter2.tree.nodes[0].subtree_depth

        # Notice Crossover protection from picking root nodes
        if min(upto1, upto2) == 1:
            depth = 1
        else:
            depth = np.random.randint(1, min(upto1, upto2))

        # Pick candidate node for crossover point in parent 1
        node1 = np.random.randint(1, len(filter1.tree.nodes))

        # ensure is a node at the selected depth
        while filter1.tree.nodes[node1].current_tree_depth != depth:

            # Pick candidate node for crossover point in parent 1
            node1 = np.random.randint(1, len(filter1.tree.nodes))


        # Pick candidate node for crossover point in parent 2
        node2 = np.random.randint(1, len(filter2.tree.nodes))

        # ensure is a node at the selected depth
        while filter2.tree.nodes[node2].current_tree_depth != depth:

            # Pick candidate node for crossover point in parent 2
            node2 = np.random.randint(1, len(filter2.tree.nodes))

        offspring1 = deepcopy(filter1)
        offspring2 = deepcopy(filter2)

        offspring1.tree, offspring2.tree = subtree_crossover(filter1.tree, node1, filter2.tree, node2)

        return offspring1, offspring2

    @staticmethod
    def composition(filter1):
        ''' This is a static method of the Simple Regressor class of individuals that defines the
        composition operation for such kind of individuals. It is basically a wrapper for compo-
        sition. It receives as input one individual, the hat function' max allowed depth and a
        grow method (optional), and return as output one new individual.'''

        g_depth=3

        grow_method='variable'

        offspring = deepcopy(filter1)

        offspring.tree = function_composition(filter1.tree, g_depth=g_depth, grow_method=grow_method)

        for i in range(1, len(offspring.tree.nodes)):
            offspring.tree.nodes[i].parent_type = 'f1'

        return offspring
