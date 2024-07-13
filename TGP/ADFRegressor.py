#-----------------------------------------------------------------------------------#
# This file provides a GP-individual that implements the concept of Automatically   #
# Defined Functions (ADF) as proposed by Koza (1994).                               #
#
# The individual herein implemented consists of 1 main function and 1 ADF. Genetic  #
# operations are implemented for this individual. The entire class is derived from  #
# SimpleRegresor.                                                                   #
#
# Extending this class to include multiple ADFs shouldn't be difficult. Similarly,  #
# this individual is defined to perform regression, but making a classification     #
# variant of it, should be easy as well, just changing the fitness functions.       #
#                                                
# This module exemplifies the power of the versatility of TurboGP: how implementing #
# advanced and complex GP variants is relatively easy, thanks to the flexibility of #
# TurboGP.                                                                          #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl et al.                                                    #
# Instituto Nacional de Astrofisica, Optica y Electronica (INAOE). Puebla, Mexico   #
# and Centro de Investigacion Cientifica y Educacion Superior de Ensenada, (CICESE) #
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#


from GPIndividuals import *
from GPOperators import *
from Regressor import *

import numpy as np


class Regressor1ADF(SimpleRegresor):
    ''' This class implements a GPtree based regressor comprised of two GP trees: one tree represents the "main function"
    that is evaluated whenever attempting to make a prediction, and another tree that represents a subroutine that can 
    be called from the main function tree. The subroutine tree is represented by a primitive within the main tree. This
    is, the subroutine tree is a primitive-defining tree that is coevolving along the main tree.'''

    def __init__(self, input_vector_size, adf_arity, complexity, adf_complexity, temp_var = 0.02, grow_method='variable'):
        '''This is the constructor method for the Regrssor1ADF class. In order to create and initialize an individual
        of this class, you have to define the size of the input vector (input_vector_size), the arity (number of input
        variables) of the Automatically Defined Function, the max tree depth of the main tree (complexity), the max depth
        of the ADF (adf_complexify), and the grow method.'''

        # the input variables for the ADF
        adf_input_set = np.arange(adf_arity)

        # the feature variables input set (i1) consist of a collection of integers.
        input_set = np.arange(input_vector_size)

        while True:
            # Initialize adf tree
            self.adf  = Tree(max_tree_depth = adf_complexity,
                            i1_set = adf_input_set,
                            grow_method = grow_method)
            # grow it
            self.adf.grow_random_tree()
            # grow it again if too shallow
            if self.adf.nodes[0].subtree_depth >= 2:
                break

        while True:
            # Initialize main tree
            self.tree = Tree(max_tree_depth = complexity,
                            i1_set = input_set,
                            grow_method = grow_method,
                            f4_set=[self.adf])
            # grow it
            self.tree.grow_random_tree()
            f4_nodes = self.tree.count_nodes_type()['f4']
            # grow it again if too shallow or no ADFs in main function
            if self.tree.nodes[0].subtree_depth >= 2 and f4_nodes > 0:
                break
        
        # Create an array with pointers to the trees. This will be useful to carry genetic ops
        # main tree will be tree 0.
        self.trees = [self.tree, self.adf]

        # Assign ADFs to main tree
        self.trees[0].update_f4_set(self.trees[1:])

        # Enforce usage of ADF
        self.force_adf = True

        # SimpleRegresor uses MSE as error measure. Finding an acceptable or optimal SimpleRegresor is a minimization
        # problem; thefore, fitness value is initialized to inf.
        self.fitness_value = float('Inf')

        # This coefficient is used for numeric mutation operation
        self.temp_var = temp_var
    
    def fitness(self, samples_matrix, labels_matrix):
        ''' This function calculates the actual fitness of the individual, for a given dataset. Regressor1ADF uses the
        fitness method of the base class (SimpleRegresor), to calculate the fitness, but it also adds a few more steps
        that involve calculating the total number of nodes in the individual and, if the option enabled, ensuring that
        the ADF is actually used by the main function, setting to Inf the fitness value (minimization problem) when no
        ADF nodes appear in the main function.'''

        # evaluate fitness using original method from base class
        super().fitness(samples_matrix, labels_matrix)
        # count number of nodes in main tree
        main_node_count = self.trees[0].count_nodes_type()
        # count number of nodes in ADF
        adf_node_count = self.trees[1].count_nodes_type()
        # count total number of nodes in individual and store it in a member method
        self.total_node_count = main_node_count['total'] + adf_node_count['total']

        # if use of ADF is mandatory
        if self.force_adf:
            if main_node_count['f4'] == 0:
                self.fitness_value = float('Inf')
        # This is a simple mechanism to ensure crossover and mutations do not erase ADFs from the population.

        return self.fitness_value

    @staticmethod
    def mutation(filter1):
        ''' This is a static method of the Regressor1ADF class that defines subtree mutation operation for such kind
        of individuals. It receives as input one individual and return as output one new individual where either the main
        tree or one of the ADFs has undergone a subtree mutation. The tree to be mutated is randomly picked, but just one 
        of the trees is mutated. '''

        # pick which tree will be mutated
        target = np.random.randint(len(filter1.trees))
        
        # pick which node will be mutated
        upto = len(filter1.trees[target].nodes)
        node = np.random.randint(upto)
        # copy individual to mutate
        offspring = deepcopy(filter1)
        # Replace target tree with its mutated version
        offspring.trees[target] = subtree_mutation(filter1.trees[target], node)

        # update offspring main function to point to its adfs
        offspring.trees[0].update_f4_set(offspring.trees[1:])
        # update main tree pointer to point to first tree in array
        # this is necessary because base class refers to self.tree to carry evaualtions
        offspring.tree = offspring.trees[0]
        # it would be similar to reassigning the ADF as shown below, although this is
        # of no particular use so far, other than keeping the pointers in check
        offspring.adf = offspring.trees[1]

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

        # pick which trees will undergo crossover
        target = np.random.randint(len(filter1.trees))

        # First offspring

        # pick node in first parent completely randomly
        node1 = np.random.randint(1, len(filter1.trees[target].nodes))
        # Get destiny node depth:
        origin_depth = filter1.trees[target].nodes[node1].current_tree_depth
        # Search for a valid subtree in second parent
        node2 = np.random.randint(1, len(filter2.trees[target].nodes))
        # Get subtree depth
        source_subtree_depth = filter2.trees[target].nodes[node2].subtree_depth
        # Verify if is a valid subtree to perform crossover so does not break max depth
        while origin_depth + source_subtree_depth > filter1.trees[target].max_tree_depth:
            # Search for a valid subtree in second offspring
            node2 = np.random.randint(1, len(filter2.trees[target].nodes))
            # Get subtree depth
            source_subtree_depth = filter2.trees[target].nodes[node2].subtree_depth

        # Now perform first crossover
        offspring1 = deepcopy(filter1)
        offspring1.trees[target], _ = subtree_crossover(filter1.trees[target], node1, filter2.trees[target], node2)

        # Second Offspring

        # pick node in second parent completely randomly
        node2 = np.random.randint(1, len(filter2.trees[target].nodes))
        # Get destiny node depth:
        origin_depth = filter2.trees[target].nodes[node2].current_tree_depth
        # Search for a valid subtree in first parent
        node1 = np.random.randint(1, len(filter1.trees[target].nodes))
        # Get subtree depth
        source_subtree_depth = filter1.trees[target].nodes[node1].subtree_depth
        # Verify if is a valid subtree to import
        while origin_depth + source_subtree_depth > filter2.trees[target].max_tree_depth:
            # Search for a valid subtree in first parent
            node1 = np.random.randint(1, len(filter1.trees[target].nodes))
            # Get subtree depth
            source_subtree_depth = filter1.trees[target].nodes[node1].subtree_depth

        # Now perform first crossover
        offspring2 = deepcopy(filter2)
        _, offspring2.trees[target] = subtree_crossover(filter1.trees[target], node1, filter2.trees[target], node2)

        # update offsprings main function to point to their adfs
        offspring1.trees[0].update_f4_set(offspring1.trees[1:])
        offspring2.trees[0].update_f4_set(offspring2.trees[1:])
        # update main tree pointera to point to first tree in array
        # this is necessary because base class refers to self.tree to carry evaualtions
        offspring1.tree = offspring1.trees[0]
        offspring2.tree = offspring2.trees[0]
        # it would be similar to reassigning the ADF as shown below, although this is
        # of no particular use so far, other than keeping the pointers in check
        offspring1.adf = offspring1.trees[1]
        offspring2.adf = offspring2.trees[1]

        return offspring1, offspring2
