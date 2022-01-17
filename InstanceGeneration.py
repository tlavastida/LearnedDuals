
import random
import numpy.random as np_random
import math

from BipartiteGraph import BipartiteGraph


# returns a random bipartite graph
# with n vertices on each side
# each edge is in the graph independently with probability p in (0,1)
def random_bipartite_graph(n,p):
    # the vertex names are 0 -> n-1 for the LHS and n -> 2n-1 for the RHS
    U = range(n)        # U = LHS
    V = range(n,2*n)    # V = RHS

    # build the edges
    E = []
    for u in U:
        for v in V:
            if random.random() <= p:
                E.append((u,v))
    
    return BipartiteGraph(U,V,E)



# returns a random complete BipartiteGraph()
def random_weighted_complete_bipartite(n,min_weight,max_weight):
    # the vertex names are 0 -> n-1 for the LHS and n -> 2n-1 for the RHS
    U = range(n)        # U = LHS
    V = range(n,2*n)    # V = RHS

    # build the edges and weights
    E = []
    W = []
    for u in U:
        for v in V:
            E.append( (u,v) )
            # choose the weight randomly
            weight = random.randrange(min_weight,max_weight+1)
            W.append(weight)

    return BipartiteGraph(U,V,E,W)



# planted matching with noise added to edges and subtracted from non-edges
# n = size of one side
# noise_level = value in (0,1), lower corresponds to more noise
# noise is geometrically distributed
def noisy_matching(n,match_val,unmatch_val,noise_level,stars=False):

    U = range(n)
    V = range(n,2*n)

    E = []
    W = []
    # build the edges and weights
    for u in U:
        for v in V:
            E.append((u,v))
            # if stars is turned on use weight 0 for all edges adjacent to n-1 and 2*n-1
            if stars and (u == n-1 or v == 2*n-1):
                W.append(0)
            else:
                # Use weight 1 for matched edges
                if u + n == v:
                    W.append(match_val + int(np_random.geometric(noise_level)))
                else:
                    W.append(unmatch_val - int(np_random.geometric(noise_level)))

    return BipartiteGraph(U,V,E,W)


# planted matching with noisy matched edges and unmatched edges
# n = size of one side
# matched edges are chosen uniformly in [match_lower,match_upper]
# unmatched edges are chosen uniformly in [unmatch_lower,unmatch_upper]
def noisy_matching_unif(n,match_lower,match_upper,unmatch_lower,unmatch_upper,stars=False):
    U = range(n)
    V = range(n,2*n)

    E = []
    W = []
    # build the edges and weights
    for u in U:
        for v in V:
            E.append((u,v))
            # if stars is turned on use weight 0 for all edges adjacent to n-1 and 2*n-1
            if stars and (u == n-1 or v == 2*n-1):
                W.append(0)
            else:
                # Use weight 1 for matched edges
                if u + n == v:
                    W.append(random.randrange(match_lower,match_upper+1))
                else:
                    W.append(random.randrange(unmatch_lower,unmatch_upper+1))

    return BipartiteGraph(U,V,E,W)



# assume that k divides n
# mean_weights is a dictionary on pairs (i,j), 0 <= i,j < k,
# var is the variance for the noise
def type_model(n,k,mean_weights,var):
    U = range(n)
    V = range(n,2*n)

    E = []
    W = []
    

    for i in range(k):
        for j in range(k):
            # weird hack to make it produce negative weights as well 
            # and handle edge cases
            center = var + mean_weights[(i,j)]
            p = 1 - var / center 
            N = int(center / p)

            for u in range(i*n//k,(i+1)*n//k):
                for v in range(n+j*n//k,n+(j+1)*n//k):    
                    E.append((u,v))
                    W.append( int(np_random.binomial(N,p) - var) ) 


    return BipartiteGraph(U,V,E,W)

# outputs a random matching M from L to R
# L and R should be the same size
def random_matching(L,R):
    M = set()
    random.shuffle(R)
    for u in L:
        v = R.pop()
        M.add((u,v))

    return M


    
# noisy matching with some modifications
def noisy_matching_geometric(n,match_val,unmatch_val,p,train=True,perturb_size=0):
    ret = None
    if train:
        ret = noisy_matching(n,match_val,unmatch_val,p,False)
    
    else:
        U = range(n)
        V = range(n,2*n)

        # Choose a subset of left side U
        L = list(random.sample(U,perturb_size))
        # L's matches under the planted matching
        R = [u+n for u in L]

        # choose a random matching from R to L
        MM = random_matching(L,R)

        #print(MM)
        E = []
        W = []
        # build the edges and weights
        for u in U:
            for v in V:
                E.append((u,v))

                if (u,v) in MM:
                    W.append(match_val + int(np_random.geometric(p)))
                    #W.append(match_val)
                elif (u,v) not in MM and u + n == v:
                    W.append(match_val + int(np_random.geometric(p)))
                else:
                    W.append(unmatch_val - int(np_random.geometric(p)))

        ret = BipartiteGraph(U,V,E,W)

    return ret





if __name__ == "__main__":

    # Code to test this file

    random.seed(353)
    np_random.seed(105)
    
    
    n = 20

    planted_M = [(u,u+n) for u in range(n)]


    p = 1/4

    match_weight = 1
    unmatch_weight = 100
    train = False
    perturb_size = 10

    G = noisy_matching_geometric(n,match_weight,unmatch_weight,p,train,perturb_size)

    print(G)

    print(G.matching_weight(planted_M))



    


    