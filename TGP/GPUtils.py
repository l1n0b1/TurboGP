#-----------------------------------------------------------------------------------#
# This file contains some necessary and useful functions for implementation of GP   #
# algorithms, such as selection mechanism (binary tournament, elitism, etc.), func- #
# tions requires for graphically visualyzing generated trees, etc.                  #
#
# Here are also defined methods required for the implementation of multi-population #
# or also known as island, models. These are migration operations, such as export,  #
# import_and_replace, etc.                                                          #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, Morales-Reyes, HJ Escalante. 2020.                       #
# Instituto Nacional de Astrofisica, Optica y Electronica (INAOE). Puebla, Mexico   #
# Development funded by CONACyT grant No. 436184, CONCyTEP grant 2019-52D, and INAOE#
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#


from copy import deepcopy
from numpy import inf

import networkx as nx
import numpy as np
import random
import heapq
import pickle

def binary_tournament(population, proportion, minimization=True):
    '''This function allows to easily use binary tournament selection method.
    Only arguments needed are a pointer to the pool from where individuals are going to be selected,
    the amount (in terms of proportion from the size of the original pool, like .5 = half individuals)
    of individuals to be selected, and if it is a minimization problem, or otherwise.
    It returns as output a new list the the selected individuals'''

    selected = []
    amount = int(len(population)*proportion)
    for _ in range(amount):
        contestant1 = random.choice(population)
        contestant2 = random.choice(population)

        if minimization==True:
            if contestant1.fitness_value < contestant2.fitness_value:
                selected.append(contestant1)
            else:
                selected.append(contestant2)
        else:
            if contestant1.fitness_value < contestant2.fitness_value:
                selected.append(contestant2)
            else:
                selected.append(contestant1)

    return selected

def tournament(population, proportion, minimization=True, n=20):
    '''Generalized version of tournament selection method.
    Only arguments needed are a pointer to the pool from where individuals are going to be selected,
    the amount (in terms of proportion from the size of the original pool, like .5 = half individuals)
    of individuals to be selected, if it is a minimization problem (or otherwise); the size of the
    tournament need to be defined manually defined here, because population dynamics cannont access
    such parameter for modification.
    It returns as output a new list the the selected individuals'''

    selected = []
    amount = int(len(population)*proportion)
    for _ in range(amount):
        subset_pop = []
        for _ in range(n):
            subset_pop.append(random.choice(population))

        selected.append(elite_selection(population=subset_pop, amount=1, minimization=minimization)[0])


    return selected


def fitness_proportional(population, proportion, minimization=True):
    '''This function allows to use fitness proportional selection method.
    Same as binary tournament; it receives as input a population, the amount of individuals to be selected, and
    if it is a minimization problem (yes by default). It returns as output a pool (list) of the selected
    individuals.'''

    selected = []
    amount = int(len(population)*proportion)

    # Build array of fitnesses of every individual in population
    individual_fitnesses = []
    for indivdual in population:
        individual_fitnesses.append(indivdual.fitness_value)

    a = np.asarray(individual_fitnesses)

    # Check this out..
    if minimization==True:
        # Turn minimization problem into maximization
        a = 1.0/a
        #a[a == inf] = 1

    global_fitness = a.sum()
    prob_vector = a/global_fitness


    indices = np.random.choice(a=len(population), size=amount, replace=False, p=prob_vector)
    for index in indices:
        selected.append(population[int(index)])

    return selected


def elite_selection(population, amount, minimization=True):
    '''This function allows to select the n (amount) top performers from a pool (population)
    of individuals. Unlike bt and fp functions, amount is defined as an integer of the actual
    number of top individuals desired. This is so, because this is the conventional way to
    define the amount of elitism used, being usually one or two. It returns as output a list
    with such high perfomers.'''

    selected = []

    # Build array of fitnesses of every individual in population
    individual_fitnesses = []
    for indivdual in population:
        individual_fitnesses.append(indivdual.fitness_value)

    a = np.asarray(individual_fitnesses)

    if minimization==True:
        indices = heapq.nsmallest(amount, range(len(a)), a.take)
    else:
        indices = heapq.nlargest(amount, range(len(a)), a.take)


    for index in indices:
        selected.append(population[int(index)])

    return selected

def find_worst(population, amount, minimization=True):
    '''This function is kind of the analogous of the elite_selection function, but instead of
    searching for the top performers, it searches for the n (amount) worst. amount variable
    should be an integer. This function is used in multipopulation EC schemes, where is necessary
    to find the individuals that will get replaced by migration operation. Ultimately, this
    function is exactly the same as elite_selection, except that maximization problems are
    inverted, instead of minimization ones. Also, this function does not returns the individuals,
    but their indices (positions) in the population (so they can be replaced by the migration
    operation); that is the major difference with elite_selection.'''

    # Build array of fitnesses of every individual in population
    individual_fitnesses = []
    for indivdual in population:
        individual_fitnesses.append(indivdual.fitness_value)

    a = np.asarray(individual_fitnesses)

    if minimization==False:
        indices = heapq.nsmallest(amount, range(len(a)), a.take)
    else:
        indices = heapq.nlargest(amount, range(len(a)), a.take)


    return indices


def migration(Population1, Population2, amount, minimization=True):
    '''This is the migration operation for CENTRALIZED Multi Population GPs. It receives as input
    two populations (origin and destiny) and the number of individuals to copy. This operation
    is not in the GPOperators definitions because it depends on elite_selection and find_worst functions
    which are declared here, even though migration is considered a GP operation in the literature.'''

    migrants = []

    # Find individuals that will sail from Pop1
    selected_origin= elite_selection(population=Population1, amount=amount, minimization=minimization)

    # Make hard copies of them
    for migrant in selected_origin:
        migrants.append(deepcopy(migrant))

    # Find those that will get replaced
    to_replace = find_worst(population=Population2, amount=amount, minimization=minimization)

    for i in range(len(to_replace)):
        Population2[to_replace[i]] = migrants[i]

    return None


def migration_random(Populations, amount, minimization=True):
    '''This is the migration operation for CENTRALIZED Multi Population GPs for a random topology.
    It receives as input n populations and the number of individuals to copy. This operation
    is not in the GPOperators script because it depends on elite_selection and find_worst functions.'''

    # First find, (deep) copy and group the sets of individuals that will be transfered
    # from each population.
    Migrants_sets = []

    for i in range(len(Populations)):
        migrants = []

        # Find them
        selected = elite_selection(population=Populations[i], amount=amount, minimization=minimization)

        # Copy them
        for migrant in selected:
            migrants.append(deepcopy(migrant))

        # Group them
        Migrants_sets.append(migrants)

    # Secondly, trace destiny routes for sailship of migrants
    dest = np.arange(len(Populations))
    np.random.shuffle(dest)
    repeated = 0
    for i in range(len(dest)):
        if dest[i] == i:
            repeated += 1
    while repeated > 1:
        np.random.shuffle(dest)
        repeated = 0
        for i in range(len(dest)):
            if dest[i] == i:
                repeated += 1

    # Finally, insert the selected individuals into their corresponding destinies by
    # replacing worst individuals

    for i in range(len(Populations)):

        # Find those that will get replaced
        to_replace = find_worst(population=Populations[dest[i]], amount=amount, minimization=minimization)

        # Replace them
        for j in range(len(to_replace)):
            Populations[dest[i]][to_replace[j]] = Migrants_sets[i][j]

    # Just for debugging purposes
    return Migrants_sets, dest


def export(Population, amount, target, ID=None, minimization=True):
    '''This is part of the migration operation for DECENTRALIZED Multi Population GPs. It selects
    the individuals from the population that will migrate to another population and packages them
    into an data structure along a random identifier, then get saved to disk in a file named as
    the destiny population. The random identifer helps the destiny population to detect if it is a
    a new shipment of individuals that needs to import or not (because it may be the case that des-
    tiny population is elapsing generations far more quickly than origin population(s)).'''

    np.random.seed()

    if ID is None:
        ID = np.random.randint(10000)

    migrants = elite_selection(population=Population, amount=amount, minimization=minimization)

    Package = {'ID': ID,
               'Content': migrants}

    if ID == 'Final':
        pickle.dump(Package, open("Population-{}f".format(target), "wb"), protocol=2)
    else:
        pickle.dump(Package, open("Population-{}".format(target), "wb"), protocol=2)

    return ID

def import_and_replace(Population, this_target, last_id, minimization=True):
    '''This is the other half of the migration operation for DECENTRALIZED Multi Population GPs.
    This function reads a file that ends with the this_target id from where it loads a dictionary
    that is composed of an ID and the set of individuals to import. It first compares the included
    ID with the last_id variable to corroborate that this particular set of individuals have not
    been loaded yet, and if it is the case, then loads the individuals and incorporates them into
    its Population.'''

    # Read file
    Ship = pickle.load(open("Population-{}".format(this_target), "rb"))

    # Load file Timestamp ID
    Ship_ID = Ship['ID']

    # Verify if this file has not been loaded yet
    if Ship_ID == last_id:
        # if it has, finish and return false
        return False

    # Otherwise, load individuals
    immigrants = Ship['Content']

    # Find those that will get replaced in local population
    to_replace = find_worst(population=Population, amount=len(immigrants), minimization=minimization)

    # Replace them
    for i in range(len(to_replace)):
        Population[to_replace[i]] = immigrants[i]

    return Ship_ID



def features_histogram(tree):
    ''' This function recieves as input a tree, and counts how many times each feature variable is used (appears in leaf
    nodes) by the function represented by the tree. It returns as output a vector from which an histogram can be plot.
    This is useful to study models generated, by examining on which ones, and how many times, each feature are relying
    on, in order to make predictions; this specially important in high dimensional problems, where GPs can converge
    to models that make use of very few features, or when attempting to build ensemble models that work on varying
    the input feature space for each predictor.'''

    result = np.zeros(len(tree.i1_set))

    for node in tree.nodes:
        if node.node_type == 'i1':
            result[node.function] +=1

    return result



def get_graph(tree, mode='type'):
    ''' This function receives as input a tree, and returns as output a networkx' graph, that can be plot with graphviz,
    and pygraphviz libraries.'''

    node_type = ['f1', 'f2', 'f3', 'i1', 'i2', 'i3', 'i4']
    colors = ['dodgerblue', 'pink', 'red', 'green', 'lightgray', 'lime', 'gray']

    colors_dict = {node_type[i]: colors[i] for i in range(len(node_type))}

    nodes = []
    labels = {}
    edges = []
    nodes_type = []

    for node in tree.nodes:
        nodes.append(node.node_id)
        for children in node.children_id:
            #a = (node.node_id, tree.nodes[children].node_id)
            a = (tree.nodes[children].node_id, node.node_id)
            edges.append(a)
        if node.node_type == 'i2':
            labels[node.node_id] = '{:.3f}'.format(node.function)
        else:
            labels[node.node_id] = '{}'.format(node.function)
        nodes_type.append(node.node_type)

    if mode == 'type':
        color_map = [colors_dict[nodes_type[i]] for i in range(len(nodes_type))]
    else:
        color_map = []
        for node in tree.nodes:
            if node.color == None:
                color_map.append('red')
            else:
                color_map.append(node.color)

    graph = nx.OrderedDiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    #graph2 = graph.reverse()

    return graph, labels, color_map
