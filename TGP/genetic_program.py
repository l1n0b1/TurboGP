#-----------------------------------------------------------------------------------#
# This file implements the classes and functions that provide scikit-alike objects, #
# CLI support and automatically deployment of distributed, multipopulation, GP runs.#
# This file consists of three classes and one main function:                        #
# - GeneticProgram class. Provides a scikit-alike interface to train/evolve models. #
# - GeneticProgramIE class. Same as GeneticProgram, but with support for migration  #
#   operations (for multipopulation GPs)                                            #
# - GeneticProgramD class. Allows to launch multiple GeneticProgramIE instances in  #
#   parallel that perform exchange of individuals, i.e. this class provides support #
#   for distributed, parallel, GP runs; all wrapped in the same scikit-alike object #
#   as GeneticProgram and GeneticProgramIE, for almost direct replacement.          #
# - main function. Allows to launch TurboGP instances from the command line, without#
#   the need of a python interactive shell. Works by reading a JSON configuration   #
#   file and a dataset stored in a pickle format (specified in the JSON file). Works#
#   by launching a GeneticProgramIE instance. GeneticProgramD relies on this method #
#   and works by launching multiple GP instances from the OS command line, in order #
#   to avoid the Python interpreter lock-in and allow true parallelism.             #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, Morales-Reyes, HJ Escalante. 2020.                       #
# Instituto Nacional de Astrofisica, Optica y Electronica (INAOE). Puebla, Mexico   #
# Development funded by CONACyT grant No. 436184, CONCyTEP grant 2019-52D, INAOE and#
# Cero-Uno Electronics.                                                             #
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#


# Required Python libraries
import os
import sys
os.environ['MKL_NUM_THREADS'] = '1'                         # For Intel Math Kernel Library
#os.environ['OPENBLAS_NUM_THREADS'] = '1'                   # For OpenBLAS
# if unsure which one you are using, type this command in a python terminal, after importing numpy.
#numpy.__config__.show()

import numpy as np

import time
import json
import datetime
import importlib
import subprocess
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from tqdm import tqdm
#from tqdm._utils import _term_move_up

# TurboGP libraries
from GPIndividuals import *
from GPOperators import *
from GPflavors import *
from LowLevel import *
from Mezzanine import *
from HighLevel import *
from Trimmers import *
from GPUtils import *
from Utils import *

# This is useful for when there is overflow, so output cells do not fill with warning messages.
# Disable to debug
import warnings
warnings.simplefilter('ignore')


class GeneticProgram:
    ''' This class provides the scikit-alike interface to easily generate predictors using TurboGP library.
    This class can instantiate objects that can perform functions such as fit, predict and score.'''

    def __init__(self, individual_class, ind_params, operations, operations_prob, operations_arity, pop_size=1000, generations=50, epochs=1, pop_dynamics="Steady_State", online=True, minimization=True, sel_mechanism=binary_tournament, n_jobs=1, grow_method='variable'):
        '''This is the constructor method for GeneticProgram objects. It receives as input the usual
        requiered parameters for a GP run:
        - individual_class (class), the type of individual to evolve (e.g. SimpleRegresor, BinaryClassifier, etc.)
        - ind_params (dictionary) the parameters that configure the individual to evolve (size of samples, max tree depth, etc.)
        - operations (list) the genetic operations to perform
        - operations_prob (list) the probabilities of each genetic operation to perform (in the same order as declared in list of operations)
        - operations_arity (list) the _arity_ i.e. number of parents, each genetic operation uses to generate offspring (in the same order as declared)
        - pop_size (int) population size. Default = 1000
        - generations (int) generations (ignored if online mode). Default = 50
        - epochs (int) epochs (ignored if online is false). Default = 1
        - pop_dynamics (string) name of population dynamics to use (e.g. steady_state, generational). Needs to be defined in GPflavors. Default 'steady_state'
        - online (boolean) if use online learning. Default = True
        - minimization (boolean) if problem (fitness function) is to be minimized or maximized. Default = True
        - sel_mechanism (function) function to select parents. Needs to be defined in GPUtils. Default = binary_tournament
        - n_jobs (int) if individual evaluation parallelism is to be used. Defines number of CPUs to use. Default = 1(single thread)
        - grow_method (string) Method to generate initial population (full, random, ramped, variable). Default = variable'''

        self.individual_class=individual_class
        self.ind_params=ind_params
        self.operations=operations
        self.operations_prob=operations_prob
        self.operations_arity=operations_arity

        self.generations=generations
        self.epochs=epochs

        self.online=online
        self.minimization=minimization
        self.sel_mechanism=sel_mechanism
        self.n_jobs=n_jobs


        self.diversity = []
        self.fitness = []
        self.test_fitness = []


        single_thread_dynamics = {"Steady_State": Steady_State, "Cellular": Cellular, "Recombinative_Hill_Climbing": RHC}
        parallel_dynamics = {"Steady_State": Steady_StateMP, "Cellular": CellularMP, "Recombinative_Hill_Climbing": RHCMP}
        if n_jobs == 1:
            self.pop_dynamics=single_thread_dynamics[pop_dynamics]
        else:
            self.pop_dynamics=parallel_dynamics[pop_dynamics]


        self.Population = []

        if grow_method=='variable':
            for _ in range(pop_size):
                self.Population.append(individual_class(**ind_params))
        else:
            print('HERE GOES RAMPED HALF AND HALF, FULL, RANDOM, ETC.')


    def fit(self, X, y, X_test=None, y_test=None):
        '''This function performs a complete GP run, according to the parameters setup when instantiating the object;
        that is, completes the number of evolutionary cycles established by generations or epochs variables, depending
        if online learning is enabled or not; When not online learning, _generation_ number of evolutionary cycles are
        performed, when online learning, then the number of cycles to elapse is epochs * size of batches. This method
        receives as input a two datasets, one of training and one for testing; testing dataset is optional:
        - X (numpy array) set of intances inputs from training dataset
        - y (numpy array) set of labels for the input instances for the training dataset
        - X_test (numpy array) Optional. Set of intances inputs from testing dataset.
        - y_test (numpy array) Optional. set of labels for the input instances for the testing dataset
        If testing samples are provided, then evaluation with out-of-the-bag samples is performed every generation.'''


        if self.online:
            batchesX = X
            batchesY = y

            self.no_batches = len(batchesX)
            self.generations = self.no_batches * self.epochs

            #initial evaluation
            start = time.perf_counter()
            for individual in self.Population:
                individual.fitness(batchesX[-1], batchesY[-1])
            end = time.perf_counter()
            print("First evaluation time cost: ", datetime.timedelta(seconds=end-start), flush=True)

            pbar = tqdm(total=self.generations)

            for e in range(self.epochs):

                for j in range(self.no_batches):

                    self.Population, d, bf, tf = self.pop_dynamics(Population = self.Population,
                                                                   batch = batchesX[j],           # different minibatch each cycle
                                                                   labels = batchesY[j],
                                                                   test_batch = X_test,
                                                                   test_labels = y_test,
                                                                   l_rate = 1.0,
                                                                   oper = self.operations,
                                                                   oper_prob = self.operations_prob,
                                                                   oper_arity = self.operations_arity,
                                                                   minimization = self.minimization,
                                                                   sel_mechanism = self.sel_mechanism,
                                                                   online = True,
                                                                   pro = self.n_jobs)

                    self.diversity.append(d)
                    self.fitness.append(bf)
                    self.test_fitness.append(tf)

                    self.model_ = elite_selection(population=self.Population, amount=1, minimization=self.minimization)[0]

                    pbar.set_postfix({'Training Fitness': bf})
                    pbar.update(1)

            pbar.close()


        else:
            x_training = X
            y_training = y

            #initial evaluation
            start = time.perf_counter()
            for individual in self.Population:
                individual.fitness(x_training, y_training)
            end = time.perf_counter()
            print("First evaluation time cost: ", datetime.timedelta(seconds=end-start), flush=True)

            pbar = tqdm(total=self.generations)

            for i in range(self.generations):

                self.Population, d, bf, tf = self.pop_dynamics(Population = self.Population,
                                                               batch = x_training,
                                                               labels = y_training,
                                                               test_batch = X_test,
                                                               test_labels = y_test,
                                                               l_rate = 1.0,
                                                               oper = self.operations,
                                                               oper_prob = self.operations_prob,
                                                               oper_arity = self.operations_arity,
                                                               minimization = self.minimization,
                                                               sel_mechanism = self.sel_mechanism,
                                                               online = False,
                                                               pro = self.n_jobs)

                self.diversity.append(d)
                self.fitness.append(bf)
                self.test_fitness.append(tf)

                self.model_ = elite_selection(population=self.Population, amount=1, minimization=self.minimization)[0]

                pbar.set_postfix({'Training Fitness': bf})
                pbar.update(1)

            pbar.close()


        print("Training fitness of best individual found: ", self.fitness[-1])
        if X_test:
            print("Testing  fitness of best individual found: ", self.test_fitness[-1])

        return self.model_


    def predict(self, X):
        '''This method applies the model generated to a set of samples. It follows the same behavior as predict methods
        in scikit-learn: it must receive as input a set of samples, and returns as output a vector with the same size
        of the input, that contains the set of predictions; if providd with a single sample, it may behave in unexpected
        ways or throw error.'''

        Y = []

        #TODO USE numpy vectorization
        for sample in X:
            Y.append(self.model_.predict(sample))

        return np.asarray(Y)


    def natural_score(self, X, y):
        ''' This function performs the score function, as defined by the individual class evolved, given a set of samples
        and their labels. In scikit learn this method usually returns a R^2 score, however, in this aspect TurboGP differs
        from the scikit-alike interface, because this function is defined by each class of individual, son for regressors
        may consist of some distance metric, while for classifiers it may be some accuracy metric, etc.
        - X (numpy vector) samples
        - y (numpy vector) labels
        Returns real valued score according to metric defined by individual evolved.'''

        return self.model_.score(X,y)

    @staticmethod
    def set_primitives(lowlevel, mezzanine=[], highlevel=[], constants_range=[-1.0,1.0], trimmers=[]):
        ''' This function is used to set the primitives for GeneticProgram objects. It receives as input:
        - lowlevel (list) list of strings that reference to the low level primitives as defined in LowLevel file
        - mezzanine (list) ...
        - highLevel (list) ...
        - constants_range, list of two integers that define the range of constants from which leave nodes may
          take a value (when not feature variables)
        - trimmers (list) list of trimmers, the same as for lowlevel, mezzanine and highlevel sets.'''

        Node.f1_set=lowlevel
        Node.f2_set=mezzanine
        Node.f3_set=highlevel
        Node.i2_set=constants_range
        Node.i3_set=trimmers


class GeneticProgramIE(GeneticProgram):
    '''This class descends directly (inherits) from the GeneticProgram class, and performs the same functionalty
    in the sense that provides a scikit-alike interface to launch TurboGP runs, with one major difference: GP runs
    configured and launched through instances of this class have the the ability to perform migration operations
    (import and export), therefore it is meant to implement and deploy island-based, multipopulation, GP schemes.'''

    def __init__(self, individual_class, ind_params, operations, operations_prob, operations_arity, no_populations, this_population, every_gen=10, top_percent=.1, topology='dynamic_random', pop_size=1000, generations=50, epochs=1, pop_dynamics="Steady_State", online=True, minimization=True, sel_mechanism=binary_tournament, n_jobs=1, grow_method='variable'):
        ''' Overloads parent class in order to accomodate five new setup parameters:
        - no_populations (int) Number of populations in a multipopulation GP run. This value is needed in order
          to know to which other populations invididuals can be exported to during the run; if setup to 1, then
          the whole process will behave no different than a GeneticProgram instance
        - pop_num (int) In a multipopulation scheme, what is the ID number of this population (island)
        - every_gen (int) Number of generations that must elapse everytime this population sends individuals to
          another population, e.g. 1, 5, 10, 20... etc. Default to 10
        - top_percent (float) percentage of elite individuals that will be exported , e.g .1 (10%) in a 1000
          individuals population, means 100 individuals will send copies of themselves to other population. Default, .1
        - topology (string) defines the network of migrations between populations, e.g. "ring"  will make each
          population send individuals to the population with the immediate next number of population ID. Defaul 'dynamic_random'
          '''



        GeneticProgram.__init__(self,
                                individual_class=individual_class,
                                ind_params=ind_params,
                                operations=operations,
                                operations_prob=operations_prob,
                                operations_arity=operations_arity,
                                pop_size=pop_size,
                                generations=generations,
                                epochs=epochs,
                                pop_dynamics=pop_dynamics,
                                online=online,
                                minimization=minimization,
                                sel_mechanism=sel_mechanism,
                                n_jobs=n_jobs,
                                grow_method=grow_method)

        self.no_populations = no_populations
        self.pop_num = this_population
        self.every_gen = every_gen
        self.amount = int(top_percent * pop_size)
        self.topology = topology


    def fit(self, X, y, X_test=None, y_test=None):
        ''' Same as in parent class GeneticProgram, but will perform export and import operations every_gen number
        of generations.

        Export generation generates a file on disk that contains the number of elite individuls configured by top_
        percent variable. The name of the file will be Population-#, where # stands for the ID number of the
        population the export is meant for (according to the topology desired). In the same way, Import operation
        will attempt to read a file named Population-#, where # corresponds to this_population variable, to load
        individuals sent from another island.

        Once the evolutionary run is completed, i.e. all generations have elapsed, this function will store the
        best individual found in a file called Population-#f, where # in this case will correspond to this_population
        value. This file is required to retrive the resulting predictor when TurboGP is launched in distributed
        enviroments or from the OS shell.'''


        if self.online:
            batchesX = X
            batchesY = y

            self.no_batches = len(batchesX)
            self.generations = self.no_batches * self.epochs

            #initial evaluation
            start = time.perf_counter()
            for individual in self.Population:
                individual.fitness(batchesX[-1], batchesY[-1])
            end = time.perf_counter()
            print("First evaluation time cost: ", datetime.timedelta(seconds=end-start), flush=True)
            # Export to myself so a Shipping file is generated and no error is generated if when reaching first
            # valid importing generation there is still no shipment from other population ready.
            export(Population=self.Population, amount=self.amount, target=self.pop_num, ID=0, minimization=self.minimization)

            pbar = tqdm(total=self.generations)

            import_enabled = False
            last_ID = 0
            i = 0
            for e in range(self.epochs):

                for j in range(self.no_batches):

                    self.Population, d, bf, tf = self.pop_dynamics(Population = self.Population,
                                                                   batch = batchesX[j],           # different minibatch each cycle
                                                                   labels = batchesY[j],
                                                                   test_batch = X_test,
                                                                   test_labels = y_test,
                                                                   l_rate = 1.0,
                                                                   oper = self.operations,
                                                                   oper_prob = self.operations_prob,
                                                                   oper_arity = self.operations_arity,
                                                                   minimization = self.minimization,
                                                                   sel_mechanism = self.sel_mechanism,
                                                                   online = True,
                                                                   pro = self.n_jobs)

                    self.diversity.append(d)
                    self.fitness.append(bf)
                    self.test_fitness.append(tf)

                    self.model_ = elite_selection(population=self.Population, amount=1, minimization=self.minimization)[0]

                    pbar.set_postfix({'Island': self.pop_num, 'Training Fitness': bf})
                    pbar.update(1)
                    
                    
                    # When reach a generation valid for exporting
                    if i%self.every_gen == 0 and i != 0:
                        # Export best so far
                        self.export_to()
                        # Enable import
                        import_enabled = True

                    # Check every Generation if conditions for import are met
                    if self.topology is not None:
                        if import_enabled is True:
                            # Attempt importing
                            last_ID, import_enabled = self.try_import(last_ID)
                    
                    # Increment generation counter
                    i += 1


            pbar.close()


        else:
            x_training = X
            y_training = y

            #initial evaluation
            start = time.perf_counter()
            for individual in self.Population:
                individual.fitness(x_training, y_training)
            end = time.perf_counter()
            print("First evaluation time cost: ", datetime.timedelta(seconds=end-start), flush=True)
            # Export to myself so a Shipping file is generated and no error is generated if when reaching first
            # valid importing generation there is still no shipment from other population ready.
            export(Population=self.Population, amount=self.amount, target=self.pop_num, ID=0, minimization=self.minimization)

            pbar = tqdm(total=self.generations)

            import_enabled = False
            last_ID = 0
            for i in range(self.generations):

                self.Population, d, bf, tf = self.pop_dynamics(Population = self.Population,
                                                               batch = x_training,
                                                               labels = y_training,
                                                               test_batch = X_test,
                                                               test_labels = y_test,
                                                               l_rate = 1.0,
                                                               oper = self.operations,
                                                               oper_prob = self.operations_prob,
                                                               oper_arity = self.operations_arity,
                                                               minimization = self.minimization,
                                                               sel_mechanism = self.sel_mechanism,
                                                               online = False,
                                                               pro = self.n_jobs)

                self.diversity.append(d)
                self.fitness.append(bf)
                self.test_fitness.append(tf)

                self.model_ = elite_selection(population=self.Population, amount=1, minimization=self.minimization)[0]

                pbar.set_postfix({'Island': self.pop_num, 'Training Fitness': bf})
                pbar.update(1)
                
                # When reach a generation valid for exporting
                if i%self.every_gen == 0 and i != 0:
                    # Export best so far
                    self.export_to()
                    # Enable import
                    import_enabled = True

                # Check every Generation if conditions for import are met
                if self.topology is not None:
                    if import_enabled is True:
                        # Attempt importing
                        last_ID, import_enabled = self.try_import(last_ID)


            pbar.close()


        print("Training fitness of best individual found: ", self.fitness[-1])
        if X_test:
            print("Testing  fitness of best individual found: ", self.test_fitness[-1])

        # Save best model to disk:
        export(Population=self.Population, amount=1, target=self.pop_num, ID='Final', minimization=self.minimization)

        return self.model_
    
    
    
    def export_to(self):
        
        # Pick target for destination:
        if self.topology=='dynamic_random':
            dest = np.random.randint(self.no_populations)
            # Check not to send them to this population
            while dest == self.pop_num:
                dest = np.random.randint(self.no_populations)
        elif self.topology=='linear_ring':
            dest = self.pop_num + 1
            if dest == self.no_populations:
                dest = 0
        else:
            # Unkown topology; export to an unreachable island
            dest = -1
            
        # Send them if migration enabled (if topology is not None)
        if self.topology is not None:
            ID_ex = export(self.Population, self.amount, dest, minimization=self.minimization)
            # Report
            #print("Exported to destiny: ", dest)
            #print("With Random ID: ", ID_ex)

    
    def try_import(self, last_ID):
        
        try:
            # Attempt import
            last_ID = import_and_replace(self.Population, self.pop_num, last_ID, minimization=self.minimization)
        except EOFError:
            # avoid IO collision
            time.sleep(1)
            last_ID = import_and_replace(self.Population, self.pop_num, last_ID, minimization=self.minimization)
        finally:
            # If succesfully imported, disable import until next valid generation for importing
            if last_ID is not False:
                import_enabled = False
                # Report
                #print("Imported with ID: ", last_ID)
            else:
                import_enabled = True
        
        return last_ID, import_enabled
            


class GeneticProgramD(GeneticProgram):
    ''' This class implements serves as a drop-in replacement for GeneticProgram objects for the deployment
    of multipopulation, also known as island, model GP. Even though it inherits GeneticProgram class, some
    of its functions diverge greatly from implementations found in the parent class, because how it actually
    works is by launching through the OS shell multiple TurboGP instances; each instance consisting of a
    separate population, that exchange individuals through disk operations. Once all GP instances finish,
    the best individual from each population is collected and the best one is returned. '''

    def __init__(self, individual_class, ind_params, operations, operations_prob, operations_arity, pop_size=1000, generations=50, epochs=1, pop_dynamics="Steady_State", online=True, minimization=True, sel_mechanism=binary_tournament, n_jobs=1, no_populations=4, every_gen=10, top_percent=.1, topology='dynamic_random', grow_method='variable'):
        ''' This is the constructor method for GeneticProgramD class. It takes as input almost the same parameters
        than GeneticProgramIE, but it does instantiate or call any TurboGP object or function; instead, it saves
        all parameters into JSON files, which are later used to call from OS command line TurboGP and launch separate
        GP processes.

        All parameters are the same as in GeneticProgramIE, except for one that is actually missing, this_population.
        This is due to the fact that this class will launch all separate populations, i.e. there is no need to specify
        ID population here. '''

        self.individual_class=individual_class
        self.ind_params=ind_params
        self.operations=operations
        self.operations_prob=operations_prob
        self.operations_arity=operations_arity
        self.pop_size=pop_size
        self.generations=generations
        self.epochs=epochs
        self.pop_dynamics=pop_dynamics
        self.online=online
        self.minimization=minimization
        self.sel_mechanism=sel_mechanism
        self.n_jobs=n_jobs
        self.no_populations = no_populations
        self.every_gen = every_gen
        self.top_percent = top_percent
        self.topology = topology
        self.grow_method=grow_method


        self.params_all = []

        # This is the set of parameters that are shared across all sub-populations(islands)
        for i in range(self.no_populations):
            run_params = {'dataset': "temp_ds.npz",
                          'lowlevel': self.f1_set,
                          'ind_module': self.individual_class.__module__,
                          'ind_name': self.individual_class.__name__,
                          'ind_params': self.ind_params,
                          'oper': [x.__name__ for x in self.operations],
                          #'oper_prob': ~,
                          'oper_arity': self.operations_arity,
                          'pop_size': self.pop_size,
                          'generations': self.generations,
                          'epochs': self.epochs,
                          'pop_dynamics': self.pop_dynamics,
                          'online': self.online,
                          'minimization': self.minimization,
                          'sel_mechanism': self.sel_mechanism.__name__,
                          'n_jobs': self.n_jobs,
                          'no_populations': self.no_populations,
                          'this_population': i,
                          'every_gen': self.every_gen,
                          'top_percent': self.top_percent,
                          'topology': self.topology,
                          'grow_method': self.grow_method
                          }
            self.params_all.append(run_params)

        # This is an example of a parameter (operations probabilities) that may or may not vary across different Populations.
        # This block of code may serve as a template to convert same-all parameters to each-population ones, e.g.
        # The dataset filename could be defined downhere similarly to the operations probabilities in order to attempt
        # ensemble learning.

        if type(self.operations_prob[0]) == list:
            for i in range(self.no_populations):
                self.params_all[i]['oper_prob'] = self.operations_prob[i]
        else:
            for i in range(self.no_populations):
                self.params_all[i]['oper_prob'] = self.operations_prob

        # Optional parameters

        if self.f2_set is not None:
            for i in range(self.no_populations):
                self.params_all[i]['mezzanine'] = self.f2_set
                self.params_all[i]['trimmers'] = self.i3_set


        # Once all parameters are defined, the parameters files are written to disk.
        for i in range(self.no_populations):
            with open('params_pop_{}'.format(i), "w") as fout:
                fout.write(json.dumps(self.params_all[i]))


        self.diversity = []
        self.fitness = []
        self.test_fitness = []


    def fit(self, X, y, X_test=None, y_test=None):
        '''This function serves as a drop-in replacement for the function of the same name of GeneticProgram
        class; however, it works very differently. Rather than launching the evolutionary run, this function
        packages the provided datasets into a pickled file, and then launches TurboGP instances from the CLI
        (i.e. this python scritps calls itself). Once all instances finish their run, this functions locates
        the best individual found in each population, and subsequently selects the best among them, loads it
        into memory, so it can be returned to the Python shell or script that called this function.'''


        # Datasets are packed in TurboGP cli format
        if self.online:
            dataset = {'batchesX': X,
                       'batchesY': y,
                       'x_testing': X_test,
                       'y_testing': y_test}
        else:
            dataset = {'x_training': X,
                       'y_training': y,
                       'x_testing': X_test,
                       'y_testing': y_test}

        pickle.dump(dataset, open("temp_ds.npz", "wb"), protocol=2)

        # clean outputs from previous runs
        for i in range(self.no_populations):
            os.popen("rm Population-{}f".format(i))

        # Launch processes
        for i in range(self.no_populations):
            #p = subprocess.Popen([sys.executable, 'genetic_program.py', 'params_pop_{}'.format(i)]
            p = subprocess.Popen(['python', 'genetic_program.py', 'params_pop_{}'.format(i)],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)


        # wait for GP processes to finish
        running = [1 for _ in range(self.no_populations)]
        running = np.asarray(running)
        while running.sum() > 0:
            for i in range(self.no_populations):
                if running[i] == 1:
                    if os.path.exists("Population-{}f".format(i)):
                        print('Population {} finished'.format(i))
                        running[i] = 0
            # In order to not to saturate the CPU
            time.sleep(10)


        # Find the best generated model across all sub-populations
        best_model = pickle.load(open("Population-0f", "rb"))['Content'][0]
        for i in range(self.no_populations):
            model = pickle.load(open("Population-{}f".format(i), "rb"))['Content'][0]
            if self.minimization:
                if model.fitness_value < best_model.fitness_value:
                    best_model = model
            else:
                if model.fitness_value > best_model.fitness_value:
                    best_model = model

        self.model_ = best_model

        return self.model_


    @staticmethod
    def set_primitives(lowlevel, mezzanine=[], highlevel=[], constants_range=[-1.0,1.0], trimmers=[]):
        ''' Overloaded function from GeneticProgram.'''

        GeneticProgramD.f1_set=lowlevel
        GeneticProgramD.f2_set=mezzanine
        GeneticProgramD.f3_set=highlevel
        GeneticProgramD.i2_set=constants_range
        GeneticProgramD.i3_set=trimmers





def main(file_name):

    # EXample
    """ run_params = {
        'dataset': 'new_50_2121.npz',
        'lowlevel': ['ADD', 'SUB', 'MUL', 'DIV', 'RELU', 'MAX', 'MEAN', 'MIN', 'X2', 'SIN', 'COS', 'SQRT'],
        'ind_module': 'NonConvolutionalFilter',
        'ind_name': 'NonConvFilter',
        'ind_params': {'lateral_window_size':21, 'complexity':9},
        'oper': ['mutation', 'protected_crossover'],
        'oper_prob': [.5, .5],
        'oper_arity': [1, 2],
        'no_populations': 4,
        'this_population': 0,
        'pop_size': 250,
        'generations': 50,
        'epochs': 3,
        'every_gen': 10,
        'top_percent': .1,
        'topology': 'dynamic_random',
        'pop_dynamics': "Steady_State",
        'online': True,
        'minimization': True,
        'sel_mechanism': 'binary_tournament',
        'n_jobs': 2
        } """
    #with open("params_demo", "w") as fout:
    #    fout.write(json.dumps(run_params))

    with open(file_name, "r") as fin:
        run_params = json.loads(fin.read())

    # module = importlib.import_module('my_package.my_module')
    module = importlib.import_module(run_params['ind_module'])
    GP_class = getattr(module, run_params['ind_name'])
    ind_params = run_params['ind_params']

    sel_mechanism = globals()[run_params['sel_mechanism']]

    if 'mezzanine' in run_params:
        if run_params['mezzanine'] is not None:
            GeneticProgramIE.set_primitives(lowlevel=run_params['lowlevel'], mezzanine=run_params['mezzanine'],  trimmers=run_params['trimmers'])
        else:
            GeneticProgramIE.set_primitives(lowlevel=run_params['lowlevel'])
    else:
        GeneticProgramIE.set_primitives(lowlevel=run_params['lowlevel'])
    # TODO: constants range, high level, etc.

    oper = [getattr(GP_class, i) for i in run_params['oper']]
    oper_prob = run_params['oper_prob']
    oper_arity = run_params['oper_arity']
    
    pop_size = run_params['pop_size']
    pop_dynamics = run_params['pop_dynamics']
    minimization = run_params['minimization']
    n_jobs = run_params['n_jobs']
    
    if 'topology' in run_params and run_params['topology'] is not None:
        topology = run_params['topology']
        no_populations = run_params['no_populations']
        this_population = run_params['this_population']
        every_gen = run_params['every_gen']
        top_percent = run_params['top_percent']
    else:
        topology = None
        no_populations = 1
        this_population = 0
        every_gen = 10  
        top_percent = .01
    
    # this_population parameter can also be used by autoexperimenter (auto.py)
    if 'this_population' in run_params:
        this_population = run_params['this_population']        
        
    
    online = run_params['online']
    # Read dataset file
    f = np.load(run_params['dataset'], allow_pickle=True)
    if online:
        epochs = run_params['epochs']
        generations = None
        # Load training dataset
        samples = f['batchesX']
        labels = f['batchesY']
    else:
        epochs = None
        generations = run_params['generations']
        # Load training dataset
        samples = f['x_training']
        labels = f['y_training']
    # Load testing dataset
    x_test = f['x_testing']
    y_test = f['y_testing']
    


    GP = GeneticProgramIE(individual_class=GP_class,
                          ind_params=ind_params,
                          operations=oper,
                          operations_prob=oper_prob,
                          operations_arity=oper_arity,
                          no_populations=no_populations,
                          this_population=this_population,
                          every_gen=every_gen,
                          top_percent=top_percent,
                          topology=topology,
                          pop_size=pop_size,
                          generations=generations,
                          epochs=epochs,
                          pop_dynamics=pop_dynamics,
                          online=online,
                          minimization=minimization,
                          sel_mechanism=sel_mechanism,
                          n_jobs=n_jobs)


    start = time.time()
    # Launch it!
    # But do not calculate fitness for test set, because it can be very expensive, and single threaded, in many scenearios
    result = GP.fit(samples, labels)
    end = time.time()
    # Calculate test fitness only at the end, and only if, test set provided
    if x_test is not None:
        testscore = GP.natural_score(x_test, y_test)
        print("Pop num: ", this_population, ' testing data score: ', testscore)
    else:
        testscore = 0
    
    if 'save_results_tofile' in run_params:
        save_results_tofile = run_params['save_results_tofile']
        if save_results_tofile:
            # File must exists
            with open("Results-{}".format(this_population), "a") as resultsfile:
                print("{}, {}, {}".format(GP.fitness[-1], testscore, end-start), file=resultsfile)


if __name__ == "__main__":
    setup_file = sys.argv[1]
    main(file_name=setup_file)
