# Code for doing the training/testing "online"

import time
import random
import numpy.random as np_random
from itertools import combinations


from UtilityFunctions import print_dict_to_file, median_duals, JaccardMatching

from InstanceGeneration import type_model

from BipartiteGraph import BipartiteGraph

from MinWeightPerfectMatching import MinWeightPerfectMatching, check_certificates

from geometric_type_model_exp import setup_clusters,euclid_dist,generate_instance


def online_test(V,instance_gen,trials,tests,results_fname):

    # open the output file (CSV)
    outf = open(results_fname,'w')
    header = 'Trial,Test,Method,Setup,Optimize,Runtime,Iters\n'
    outf.write(header)

    for i in range(trials):

        print('Running trial {}'.format(i))

        dual_list = []
        learned_duals = {v:0 for v in V}

        opt_solutions = []
        for t in range(tests):
            print("\tRunning test {}".format(t))

            TestG = instance_gen()

            # test the standard method
            M,d,setup_elapsed,optimize_elapsed,iter_count = MinWeightPerfectMatching(TestG,None,True,False)

            if not check_certificates(TestG,M,d):
                print('optimality check failed for standard')

            opt_solutions.append(M)
            dual_list.append(d)

            runtime = setup_elapsed + optimize_elapsed
            line = '{}, {}, {}, {}, {}, {}, {}\n'.format(i,t,'Hungarian',setup_elapsed,optimize_elapsed,runtime,iter_count)
            outf.write(line)

            #  test learned duals
            M,d,setup_elapsed,optimize_elapsed,iter_count = MinWeightPerfectMatching(TestG,learned_duals,True,True)

            if not check_certificates(TestG,M,d):
                print('optimality check failed for learned duals')

            runtime = setup_elapsed + optimize_elapsed
            line = '{},{}, {}, {}, {}, {}, {}\n'.format(i,t,'Learned Duals',setup_elapsed,optimize_elapsed,runtime,iter_count)
            outf.write(line)

            # retrain duals
            learned_duals = median_duals(dual_list)

    outf.close()


def run_type_model_exp(s):
    random.seed(s)
    np_random.seed(s)

    # parameters
    trials = 20
    tests = 30
    
    # nodes per side
    n = 500

    # number of groups
    k = 50

    V = range(2*n)

    mean_weights = {}
    for i in range(k):
        for j in range(k):
            mean_weights[(i,j)] = int(np_random.geometric(1/250))

    print_dict_to_file(mean_weights,'../Data/Online/mean_weights.txt')

    var = 200
    def instance_gen():
        return type_model(n,k,mean_weights,var)

    results_fname = '../Data/Online/online_type_model_n_{}_k_{}_tr_{}_te_{}_v_{}.csv'.format(n,k,trials,tests,var)

    #results_fname = '../Data/Online/online_test_blah.csv'

    online_test(V,instance_gen,trials,tests,results_fname)


def run_clustering_exp(s):
    random.seed(s)
    np_random.seed(s)

    # parameters
    trials = 20
    tests = 30

    # number of nodes per side
    k = 500
    # node names
    V = range(2*k)

    # distance function to use
    dist = euclid_dist

    # number of points to load - ideally all of them
    n = 98942
    fname = 'kdd_sample.csv'
    results_fname = '../Data/Online/online_clustering_exp_kdd_k_500.csv'
    

    print('Setting up clusters')
    L,R = setup_clusters(fname,n,k)

    def instance_gen():
        return generate_instance(L,R,k,dist)

    print('Starting online test')

    online_test(V,instance_gen,trials,tests,results_fname)

    


if __name__ == '__main__':

    run_type_model_exp(0)
    #run_clustering_exp(0)








