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
import time
import subprocess

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#cmap='YlGnBu'
cmap='plasma'

def run_cells_concurrently(linear_list, runs, available_GP_threads):
    
    queue = list(range(len(linear_list)))*runs
    queue.reverse()
    
    running = []
    
    while len(queue) > 0 or len(running) > 0:
        # if there are idle cpu threads and GP instances remaining, launch next
        if available_GP_threads > 0 and len(queue) > 0:
            # pull next GP run from queue
            cell = queue.pop()
            # add GP run to list of running GPs
            running.append(cell)
            # launch it
            p = subprocess.Popen(['python', 'genetic_program.py', linear_list[cell]['filename']],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            # Reduce number of available global GP threads
            available_GP_threads -= 1
            
        # check for GP runs that have finished
        if len(running) > 0:
            for i in running:
                # check each GP in running list if finished
                if os.path.exists("Population-{}f".format(i)):
                    print('Population {} finished'.format(i))
                    # Delete finish file so there is no conflict in next repetition
                    os.remove("Population-{}f".format(i)) 
                    # remove GP run from list of running GPs
                    running.remove(i)
                    # return (increase) GP thread
                    available_GP_threads += 1
        
        # In order to not to saturate the CPU
        time.sleep(10)
    
    return None

def report_results(x_axis, y_axis, z_axis, linear_list):
    
    results = {}
    
    col_names = x_axis
    row_names = y_axis
    
    for table in z_axis:
        
        D = {}
        
        D['Training Avg'] = pd.DataFrame(columns = col_names, index = row_names)
        D['Training Std'] = pd.DataFrame(columns = col_names, index = row_names)
        D['Testing Avg'] = pd.DataFrame(columns = col_names, index = row_names)
        D['Testing Std'] = pd.DataFrame(columns = col_names, index = row_names)
        D['Time Avg'] = pd.DataFrame(columns = col_names, index = row_names)
        D['Time Std'] = pd.DataFrame(columns = col_names, index = row_names)
        
        results[table] = D
    
    for i in range(len(linear_list)):
        temp_df = pd.read_csv("Results-{}".format(i))
        
        x = linear_list[i]['x']
        y = linear_list[i]['y']
        z = linear_list[i]['z']
        
        results[z]['Training Avg'].loc[y,x] = temp_df['Training'].mean()
        results[z]['Training Std'].loc[y,x] = temp_df['Training'].std()
        results[z]['Testing Avg'].loc[y,x] = temp_df['Testing'].mean()
        results[z]['Testing Std'].loc[y,x] = temp_df['Testing'].std()
        results[z]['Time Avg'].loc[y,x] = temp_df['Time'].mean()
        results[z]['Time Std'].loc[y,x] = temp_df['Time'].std()
    
    return results

def save_heatmaps(result):
    
    for table in result:
        plt.figure()
        df_ = pd.DataFrame(result[table]['Training Avg'],dtype="float")
        sns.heatmap(df_,  xticklabels=True, yticklabels=True, cmap=cmap).set_title('{} - Training Avg'.format(table))
        plt.savefig("{} - Training Avg.png".format(table))
        
        plt.figure()
        df_ = pd.DataFrame(result[table]['Training Std'],dtype="float")
        sns.heatmap(df_,  xticklabels=True, yticklabels=True, cmap=cmap).set_title('{} - Training Std'.format(table))
        plt.savefig("{} - Training Std.png".format(table))
        
        plt.figure()
        df_ = pd.DataFrame(result[table]['Testing Avg'],dtype="float")
        sns.heatmap(df_,  xticklabels=True, yticklabels=True, cmap=cmap).set_title('{} - Testing Avg'.format(table))
        plt.savefig("{} - Testing Avg.png".format(table))
        
        plt.figure()
        df_ = pd.DataFrame(result[table]['Testing Std'],dtype="float")
        sns.heatmap(df_,  xticklabels=True, yticklabels=True, cmap=cmap).set_title('{} - Testing Std'.format(table))
        plt.savefig("{} - Testing Std.png".format(table))
        
        plt.figure()
        df_ = pd.DataFrame(result[table]['Time Avg'],dtype="float")
        sns.heatmap(df_,  xticklabels=True, yticklabels=True, cmap=cmap).set_title('{} - Time Avg'.format(table))
        plt.savefig("{} - Time Avg.png".format(table))
        
        plt.figure()
        df_ = pd.DataFrame(result[table]['Time Std'],dtype="float")
        sns.heatmap(df_,  xticklabels=True, yticklabels=True, cmap=cmap).set_title('{} -Time Std'.format(table))
        plt.savefig("{} - Time Std.png".format(table))

    
    return None

def make_experiment():
    
     # Array of experiments to perform (list of lists)
    experiment = []    
    # same list as above, but linear (a single list) for easier management
    l = []
    
    
    ###########################################################################################################################################
    
    # parameters to to vary;
    # imagine a scientific paper results table; these are the parameters to sweep,
    # x and y axis correspong to a single table, and each parameter in z is another
    # table. This implementation provides up to 3 parameters to vary.
    
    x_axis = [100, 500]
    y_axis = [1000, 2000]
    z_axis = [7,8]
    
    
    # number of trial runs per experiment performed, i.e. per cell in the table 
    # defined above. The value typically used here is "30", in order to extract
    # some 'statistically significant' results; although in reality it can be 
    # any other value that is meaningful to the study being carried.
    
    trial_runs = 15 
    
    # number of CPUs to allocate for all the experiments; this parameter will be
    # used in combination with the n_jobs parameters (no. of cpus to be used per
    # individual GP run) in order to determine how many GPs can be launched in 
    # parallel.
    
    allocated_cpus = 16
    
    # Parameters to remain static.
    # Disable the parameters that will be part of the study by commenting.
    # Remember that 'epochs' or 'generations' will be ignored, depending if 'online' true or false.
                
    run_params = {#'dataset':             'temp_ds.npz',
                  'lowlevel':            ['ADD', 'SUB', 'MUL', 'DIV', 'RELU', 'MAX', 'MEAN', 'MIN', 'X2', 'SQRT'],
                  'mezzanine':           None,
                  'trimmers':            None,
                  'ind_module':          'Regressor',
                  'ind_name':            'RegressorLS',
                  #'ind_params':          {'input_vector_size':60, 'metric':'f1_score', 'complexity':9},
                  'oper':                ['mutation', 'protected_crossover'],
                  'oper_prob':           [.5, .5],
                  'oper_arity':          [1, 2],
                  #'pop_size':            1000,
                  'online':              True,
                  #'generations':         100,
                  'epochs':              10,
                  'pop_dynamics':        "Steady_State",
                  'minimization':        True,
                  'sel_mechanism':       'binary_tournament',
                  'n_jobs':              4,
                  
                  # ### DO NOT MODIFY PARAMETERS BELOW ### #
                  # TODO: Auto does not support distributed (island model) GPs yet
                  #'no_populations': 1,
                  #'this_population': 0,
                  #'every_gen': 10,
                  #'top_percent': .1,
                  #'topology': None,
                  'grow_method': 'variable',
                  'save_results_tofile': True
                  }
    
    
    # Max number of GP processes that can be launched simultaneously
    available_GP_threads = allocated_cpus // run_params['n_jobs']
    
    this_population = 0
    for table in z_axis:
        # Parameter to sweep in tables
        run_params['ind_params']  =  {'input_vector_size':2, 'complexity':table}
        
        
        rows = []
        for row in y_axis:
            # Parameter to sweep in rows
            run_params['pop_size']  =  row           
            
            columns = []
            for column in x_axis:
                # Parameter to sweep in columns
                run_params['dataset']  =  "keijzer12-05pi-5000-{}.npz".format(column)
                
                
                
                
     ##########################################################################################################################################           
                
                
                run_params['this_population']  =  this_population
                with open('TGP_ex_{}_{}_{}'.format(table, row, column), "w") as fout:
                    fout.write(json.dumps(run_params))
                
                if os.path.exists("Results-{}".format(this_population)) is False:
                    with open('Results-{}'.format(this_population), 'w') as fp:
                        print("Training,Testing,Time", file=fp)
                
                cell = {'filename':   'TGP_ex_{}_{}_{}'.format(table, row, column),
                        'z':          table,
                        'y':          row,
                        'x':          column
                       }
                
                columns.append(cell)
                l.append(cell)
                
                this_population += 1
            
            rows.append(columns)
            
        experiment.append(rows)
        
    #run_cells_concurrently(linear_list=l, runs=trial_runs, available_GP_threads=available_GP_threads)

    return x_axis, y_axis, z_axis, l, trial_runs, available_GP_threads

def main():
    
    # generate setup file for GP runs and (if do no exists) empty files for results.
    # if files for results do exists, are not modified, so auto can be run over previous auto to add more trial runs.
    x_axis, y_axis, z_axis, linear_list, runs, available_GP_threads = make_experiment()
    
    # launch experiments using assigned CPU threads
    #run_cells_concurrently(linear_list=linear_list, runs=runs, available_GP_threads=available_GP_threads)
    
    # load results from files in disk to pandas dataframes
    result = report_results(x_axis, y_axis, z_axis, linear_list)
    
    # plot results in image files
    save_heatmaps(result)


if __name__ == "__main__":
    main()
