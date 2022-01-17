# Code for the type model experiments

import time
import random
import numpy.random as np_random
from itertools import combinations


from UtilityFunctions import print_dict_to_file, train_duals, JaccardMatching

from InstanceGeneration import type_model

from BipartiteGraph import BipartiteGraph

from MinWeightPerfectMatching import MinWeightPerfectMatching, check_certificates


# train the duals, run tests
def test_algs(duals_fname,results_fname,samples,tests,instance_gen):
    start = time.time()
    # do the training step
    median_duals = train_duals(samples,instance_gen)
    stop = time.time()

    # output the trained duals (CSV file)
    dualf = open(duals_fname,'w')
    dualf.write("V, median_dual, avg_dual\n")
    for v in median_duals.keys():
        dualf.write("{}, {}\n".format(v,median_duals[v]))
    dualf.close()

    print('Training time: {}'.format(stop-start))

    # totals to keep track of
    algs = ['standard', 'median_tighten'] 
    #,'average_tighten']
    setup_totals = {a:0 for a in algs}
    optimize_totals = {a:0 for a in algs}
    iters_totals = {a:0 for a in algs}

    # open the output file (CSV)
    outf = open(results_fname,'w')
    outf.write('trial, std setup, std optimize, std iters, median_tighten setup, median_tighten optimize, median_tighten iters\n')

    opt_solutions = []

    for t in range(tests):

        print("Running test {}".format(t))
        outf.write('{}, '.format(t))
        TestG = instance_gen()

        # test the standard method
        M,d,setup_elapsed,optimize_elapsed,iter_count = MinWeightPerfectMatching(TestG,None,True,False)

        if not check_certificates(TestG,M,d):
            print('optimality check failed for standard')

        opt_solutions.append(M)

        setup_totals['standard'] += setup_elapsed
        optimize_totals['standard'] += optimize_elapsed
        iters_totals['standard'] += iter_count

        outf.write('{}, {}, {}, '.format(setup_elapsed,optimize_elapsed,iter_count))

        # test the median_tighten method
        M,d,setup_elapsed,optimize_elapsed,iter_count = MinWeightPerfectMatching(TestG,median_duals,True,True)

        if not check_certificates(TestG,M,d):
            print('optimality check failed for median+tighten')

        setup_totals['median_tighten'] += setup_elapsed
        optimize_totals['median_tighten'] += optimize_elapsed
        iters_totals['median_tighten'] += iter_count

        outf.write('{}, {}, {}\n'.format(setup_elapsed,optimize_elapsed,iter_count))

    outf.close()

    # compute average jaccard similarity
    denom = 0
    tot = 0
    for M1,M2 in combinations(opt_solutions,2):
        tot += JaccardMatching(M1,M2)
        denom += 1
    
    print("Average Jaccard: {}".format(tot/denom),flush=True)

    return "All tests completed successfully"

def run_exps(s):
    # set up seeds
    random.seed(s)
    np_random.seed(s)

    # parameters
    samples = 20
    tests = 10
    
    # nodes per side
    n = 500

    # number of groups
    k = 50

    mean_weights = {}
    for i in range(k):
        for j in range(k):
            mean_weights[(i,j)] = int(np_random.geometric(1/250))

    print_dict_to_file(mean_weights,'../Data/type_model_exp/mean_weights.txt')

    DIR = '../Data/type_model_exp2/'
    BODY = 'type_model_s_{}_t_{}'.format(samples,tests)

    # variance values to test
    #variances = [10*j for j in range(11,36)]
    variances = [2**j for j in range(15,21)]

    for var in variances:
        duals_fname = DIR + 'duals_' + BODY + '_k_{}_v_{}.csv'.format(k,var)
        results_fname = DIR + 'results_' + BODY + '_k_{}_v_{}.csv'.format(k,var)

        def instance_gen():
            return type_model(n,k,mean_weights,var)

        test_algs(duals_fname,results_fname,samples,tests,instance_gen)
        

if __name__ == '__main__':
    run_exps(0)



