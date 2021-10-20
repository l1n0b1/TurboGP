#-----------------------------------------------------------------------------------#
# This file provides some handy functions that allow to automatize the execution of #
# multiple GP runs in a systematic fashion, with the aim of performing comparisons  #
# between different parameters setups, genetic operations performance, populations  #
# models, or any other comprehensive scientific study in general.                   #
#
# These functions rely con the TurboGP CLI provided by genetic_program.py script.   #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, et. al. 2020.                                            #
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#
import os
import sys
import json

def main():
    
    # parameters to to vary;
    # imagine a scientific paper results table; these are the parameters to sweep,
    # x and y axis correspong to a single table, and each parameter in z is another
    # table. This implementation provides up to 3 parameters to vary.
    
    x_axis = [250, 500, 1000, 2000, 4000]
    y_axis = [5, 6, 7, 8, 9]
    z_axis = ['regression.npz']
    
    
    # number of trial runs per experiment performed, i.e. per cell in the table 
    # defined above. The value typically used here is "30", in order to extract
    # some 'statistically significant' results; although in reality it can be 
    # any other value that is meaningful to the study being carried.
    
    trial_runs = 5 
    
    
    # Parameters to remain static.
    # Disable the parameters that will be part of the study by commenting.
    # Remember that 'epochs' or 'generations' will be ignored, depending 
    # if 'online' true or false.
                
    run_params = {#'dataset':             'temp_ds.npz',
                  'lowlevel':            ['ADD', 'SUB', 'MUL', 'DIV', 'RELU', 'MAX', 'MEAN', 'MIN', 'X2', 'SIN', 'COS', 'SQRT'],
                  'mezzanine':           None,
                  'trimmers':            None,
                  'ind_module':          'Classifier',
                  'ind_name':            'BinaryClassifier',
                  #'ind_params':          {'input_vector_size':60, 'metric':'f1_score', 'complexity':9},
                  'oper':                ['mutation', 'protected_crossover'],
                  'oper_prob':           [.5, .5],
                  'oper_arity':          [1, 2],
                  #'pop_size':            1000,
                  'online':              False,
                  'generations':         100,
                  'epochs':              1,
                  'pop_dynamics':        "Steady_State",
                  'minimization':        False,
                  'sel_mechanism':       'binary_tournament',
                  'n_jobs':              4,
                  
                  # ### DO NOT MODIFY PARAMETERS BELOW ### #
                  # TODO: Auto does not support distributed (island model) GPs yet
                  #'no_populations': 1,
                  #'this_population': 0,
                  #'every_gen': 10,
                  #'top_percent': .1,
                  #'topology': None,
                  'grow_method': 'variable'
                  }
    
    
    
    for table in z_axis:
        # Parameter to sweep in tables
        run_params['dataset']  =  table
        
        for row in y_axis:
            # Parameter to sweep in rows
            run_params['ind_params']  =  {'input_vector_size':60, 'metric':'f1_score', 'complexity':row}
            
            for column in x_axis:
                # Parameter to sweep in columns
                run_params['pop_size']  =  column
                 
                with open('TGP_ex_{}_{}_{}'.format(table, row, column), "w") as fout:
                    fout.write(json.dumps(run_params))
                
                

    

if __name__ == "__main__":
    main()
