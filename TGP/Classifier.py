#-----------------------------------------------------------------------------------#
# This file implements a GP-individual based on a single tree and designed to carry #
# binary classification tasks. Along the regressor, it can be considered the most   #
# basic example of GP-individuals in the TurboGP library. It is very similar to the #
# regressor individual, with the main change being how the individual is evaluared; #
# rather than using the MSE or some other distance metric as fitness function, this #
# individual uses classification metrics as objetive functions; a couple of metrics #
# are already defined and ready to be used, accuracy and F1 score, while others can #
# be easily added.                                                                  #
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


class BinaryClassifier:
    ''' This class implements a GPtree based binary classifier. In consists of a tree object (as defined in GPIndividual.py),
    a fitness value, functions that allow to perform a prediction (predict), evaluate the performance of the individual
    against a single sample and its label (case_fitness), evaluate the performance of the individual against a bunch
    of labeled samples (fitness), and methods that implement genetic operations applicable to this type of individual.'''

    def __init__(self, input_vector_size, complexity, metric='accuracy', grow_method='variable'):
        '''This is the constructor method for the BinaryClassifier class. In order to create and initialize an individual
        of this class, you have to define the size of the input vectors (input_vector_size), the max tree depth allowed
        (complexity), the classification metric used evaluate and guide the evolution of instances of this class, and the
        grow method.'''

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

        # BinaryClassifier may use different classification metrics in order to guide the evolutionary process. As such,
        # it depends on the metric used on whether individuals' fitness will be initialized to Inf (minimization) or
        # 0 (maximization).
        self.metric = metric
        if self.metric in ['accuracy', 'f1_score']:
            # maximization metrics
            self.fitness_value = 0.0
        else:
            # e.g., log loss/binary crossentropy
            self.fitness_value = float('Inf')

    def predict(self, instance):
        '''This function applies the individual to a sample (instance) in order to perform a prediction. '''

        output = self.tree.evaluate(instance)

        # thresholding
        if output > 0:
            result = 1
        else:
            result = 0

        return result

    def case_fitness(self, instance, label):
        ''' Test performance of the individual against a single instance.'''

        prediction = self.predict(instance)

        # if correct, return 1, otherwise, return 0
        if prediction == label:
            result = 1
        else:
            result = 0

        return result

    def fitness(self, samples_matrix, labels_matrix):
        ''' This function calculates the actual fitness of the individual, for a given dataset. BinaryClassifier may use
        different metrics as objective function; here to cases of guiding metrics are already defined: accuracy, and
        f1 score.'''

        batch_size = len(samples_matrix)

        true_values = labels_matrix

        predictions = []
        for i in range(batch_size):
            predictions.append(self.predict(samples_matrix[i]))

        predictions = np.array(predictions)


        if self.metric == 'accuracy':
            accuracy = (true_values == predictions).sum() / batch_size
            self.fitness_value = accuracy
        elif self.metric == 'f1_score':
            TP = ((predictions == 1) & (true_values == 1)).sum()
            FP = ((predictions == 1) & (true_values == 0)).sum()
            FN = ((predictions == 0) & (true_values == 1)).sum()
            f1 = TP / (TP + (.5 * (FP + FN)))
            self.fitness_value = f1
        else:
            self.fitness_value = 0.0

        return self.fitness_value

    def score(self, samples_matrix, labels_matrix, metric='accuracy'):
        ''' Score functions in TurboGP individuals are designed to test the performance of an individual against a set
        of samples _without_ modifying its fitness value; i.e., this functions can be used in testing and validation
        scenarios. It can also be the case that here are defined perfomance metrics that are unsuited to guide evolution;
        for example, in the case of binary classification, both precision and recall are very important and informative
        performance measure, but they cannot be used to guide an evolutionary optimization process, and both are more
        akin to used as post-training evaluation methods.'''

        batch_size = len(samples_matrix)

        true_values = labels_matrix

        predictions = []
        for i in range(batch_size):
            predictions.append(self.predict(samples_matrix[i]))

        predictions = np.array(predictions)


        if metric == 'accuracy':
            accuracy = (true_values == predictions).sum() / batch_size
            testing_fitness = accuracy
        elif metric == 'f1_score':
            TP = ((predictions == 1) & (true_values == 1)).sum()
            FP = ((predictions == 1) & (true_values == 0)).sum()
            FN = ((predictions == 0) & (true_values == 1)).sum()
            f1 = TP / (TP + (.5 * (FP + FN)))
            testing_fitness = f1
        elif metric == 'precision':
            TP = ((predictions == 1) & (true_values == 1)).sum()
            FP = ((predictions == 1) & (true_values == 0)).sum()
            precision = TP / (TP + FP)
            testing_fitness = precision
        elif metric == 'recall':
            TP = ((predictions == 1) & (true_values == 1)).sum()
            FN = ((predictions == 0) & (true_values == 1)).sum()
            recall = TP / (TP + FN)
            testing_fitness = recall
        else:
            testing_fitness = 0.0

        return testing_fitness


    @staticmethod
    def crossover(filter1, filter2):
        ''' This is a static method of the BinaryClassifier class that defines standard subtree crossover operation for
        this type of GP individuals. It is basically a wrapper for subtree_crossover method found in GPOperators.py.
        It receives as input two individuals and returns as output two new individuals. This version of
        crossover is not protected in any way, neither gramatically nor in max tree depth. It does nothing more than
        choosing random nodes in both trees as exchange points, and call subtree_crossover method on them. Therefore it
        should be used only when just low-level primitves are enabled (i.e. no mezzanine functions), and it may induce
        bloating in indivduals as generations elapse, making difficult to predict when a GP run will finish (runs may
        extend in time indefinitely).'''

        upto1 = len(filter1.tree.nodes)
        upto2 = len(filter2.tree.nodes)

        # Debiasing crossover points probabilities, so leaf nodes do not take over majority of crossovers operations
        probs1 = []
        for i in range(len(filter1.tree.nodes)):
            if i == 0:
                probs1.append(.0)
            else:
                if filter1.tree.nodes[i].node_type in ['i1', 'i2', 'i3', 'i4']:
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
                if filter2.tree.nodes[i].node_type in ['i1', 'i2', 'i3', 'i4']:
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
        ''' This is a static method of the BinaryClassifier class that defines subtree mutation operation for such kind
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
        ''' This is a static method of the BinaryClassifier class of individuals that defines a protected crossover type
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
        ''' This is a static method of the BinaryClassifier class of individuals that defines the SameDepths subtree
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
        ''' This is a static method of the BinaryClassifier class of individuals that defines the
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
